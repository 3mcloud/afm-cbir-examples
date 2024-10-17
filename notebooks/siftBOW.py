from sklearn.cluster import MiniBatchKMeans
import os
import numpy as np
from tqdm import tqdm
import cv2
import numpy as np
import os
from p_tqdm import p_map
import pandas as pd

def run_SIFT_image(image_file,max_feature_cap=500,contrast_thresh=0.01):
	test_img = cv2.imread(image_file)
	sift = cv2.SIFT_create(contrastThreshold=contrast_thresh)
	kp,des = sift.detectAndCompute(test_img,None)
	if len(des.shape)==1:
		return des
	if des.shape[1]>max_feature_cap:
		des = des[:,:max_feature_cap]
	return des

def run_SIFT_folder(img_folder,sift_npy_folder,n_cores=1,\
			max_feature_cap=500,contrast_thresh=0.01):
	#Standard img_folder is "AFM_height", standard sift_npy_folder is "SIFT_FEAT_NPY"
	if not os.path.isdir(sift_npy_folder):
		os.mkdir(sift_npy_folder)
	all_img_files = [os.path.join(img_folder,i) for i in os.listdir(img_folder)]
	def run_sift(image_file):
		try:
			test_img = cv2.imread(image_file)
			sift = cv2.SIFT_create(contrastThreshold=contrast_thresh)
			kp,des = sift.detectAndCompute(test_img,None)
			if des.shape[1]>max_feature_cap:
					des = des[:,:max_feature_cap]
			np.save(image_file.replace(img_folder,sift_npy_folder).replace('.png','__SIFT.npy'),des)
		except AttributeError:
			pass
	all_outs = p_map(run_sift,all_img_files,num_cpus=n_cores)

def cluster_BOW(sift_npy_folder,cluster_npy=None,num_clusters=32,n_epochs=5):
	#Currently held in "SIFT_FEAT_NPY"
	all_sift_npy = [os.path.join(sift_npy_folder,i) for i in os.listdir(sift_npy_folder)]
	kmeans = MiniBatchKMeans(n_clusters=num_clusters,batch_size=100)
	for _ in range(n_epochs):
		for filename in tqdm(all_sift_npy):
			X = np.load(filename)
			kmeans = kmeans.partial_fit(X)
	if cluster_npy is None:
		np.save(f'sift_bag_of_words_{num_clusters}.npy',kmeans.cluster_centers_)
	else:
		np.save(cluster_npy,kmeans.cluster_centers_)

def build_full_embedding(sift_npy_folder,cluster_npy,out_csv,n_cores=1):
	n_clusters = np.load(cluster_npy)
	all_npy_files = [os.path.join(sift_npy_folder,i) for i in os.listdir(sift_npy_folder)]
	def BOW_image_encoding_parallel(npy_file):
		descs = np.load(npy_file)
		final_hist = np.zeros(len(n_clusters))
		for i in descs:
			clust_ind = np.argmin(np.sum((n_clusters-i)**2,axis=1)**0.5)
			final_hist[clust_ind]+=1
		final_hist = np.array(final_hist,dtype=np.float32)/np.sum(final_hist)
		return final_hist
	embeddings = p_map(BOW_image_encoding_parallel,all_npy_files,num_cpus=n_cores)
	cols_li = [f'embed_{str(i).zfill(4)}' for i in range(len(n_clusters))]
	embedding_df = pd.DataFrame(embeddings,columns=cols_li)
	embedding_df['filename']=all_npy_files
	embedding_df.to_csv(out_csv,index=False)

#def BOW_image_encoding(image_file,cluster_npy):
#	n_clusters = np.load(cluster_npy)
#	descs = run_SIFT_image(image_file)
#	final_hist = np.zeros(len(n_clusters))
#	if len(descs) == 0:
#		return np.array(final_hist,dtype=np.float32)
#	for i in descs:
#		clust_ind = np.argmin(np.sum((n_clusters-i)**2,axis=1)**0.5)
#		final_hist[clust_ind]+=1
#	final_hist = np.array(final_hist,dtype=np.float32)/np.sum(final_hist)
#	return final_hist

class siftBOW_inference:
	def __init__(self,cluster_npy):
		self.n_clusters = np.load(cluster_npy)
	
	def infer(self,input_file):
		descs = run_SIFT_image(input_file)
		final_hist = np.zeros(len(self.n_clusters))
		if len(descs) == 0:
			return np.array(final_hist,dtype=np.float32)
		for i in descs:
			clust_ind = np.argmin(np.sum((self.n_clusters-i)**2,axis=1)**0.5)
			final_hist[clust_ind]+=1
		final_hist = np.array(final_hist,dtype=np.float32)/np.sum(final_hist)
		return final_hist