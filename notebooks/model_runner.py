#from optical_reverse_image_search.models import moco, simclr, dino
import dino
import argparse
import os

#models_list = ['moco','simclr','dino']
models_list = ['dino']

def model_main(model_type,in_dir,input_size,latent_size,batch_size,num_workers,epochs,model_file,embeddings_file):
    #Limiting release to just training on DINO. SimCLR and MoCo proceed with a very similar setup.
    #if model_type == 'moco':
    #    moco.MoCo_train(in_dir,input_size,latent_size,batch_size,\
    #            num_workers,epochs,model_file=model_file,embeddings_file=embeddings_file)
    #elif model_type == 'simclr':
    #    simclr.SimCLR_train(in_dir,input_size,latent_size,batch_size,\
    #            num_workers,epochs,model_file=model_file,embeddings_file=embeddings_file)
    if model_type == 'dino':
        dino.DINO_train(in_dir,input_size,latent_size,batch_size,\
                num_workers,epochs,model_file=model_file,embeddings_file=embeddings_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',type=str,help='Model architecture to run.',required=True)
    parser.add_argument('--in_dir',type=str,help='Input directory to use for training.',required=True)
    parser.add_argument('--input_size',type=int,help='Rescaled image input size.',default=128)
    parser.add_argument('--latent_size',type=int,help='Number of features in the latent space.',default=32)
    parser.add_argument('--batch_size',type=int,help='Model training batch size.',default=256)
    parser.add_argument('--num_workers',type=int,help='Number of cores to use for ingest.',default=64)
    parser.add_argument('--epochs',type=int,help='Number of epochs to use in run.',default=10)
    parser.add_argument('--model_file',type=str,help='Name for .pth file to save architecture/weights.',default='model.pth')
    parser.add_argument('--embeddings_file',type=str,help='Name for .csv file to save embeddings.',default='embeddings.csv')
    args = parser.parse_args()
    if args.model not in models_list:
        raise ValueError(f'{args.model} not in list of models {models_list}')
    if not os.path.isdir(args.in_dir):
        raise ValueError(f'{args.in_dir} is not a valid path')
    model_main(args.model,args.in_dir,args.input_size,args.latent_size,args.batch_size,\
               args.num_workers,args.epochs,args.model_file,args.embeddings_file)
