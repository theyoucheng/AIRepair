
__all__=["declare_reproducible"]

def declare_reproducible(SEED = 123):
    '''
    https://github.com/NVIDIA/framework-determinism/blob/master/pytorch.md
    '''
    # or whatever you choose
    try :
        random.seed(SEED) # if you're using random
    except :
        pass 

    try :
        np.random.seed(SEED) # if you're using numpy
    except :
        pass 

    try :
        torch.manual_seed(SEED) # torch.cuda.manual_seed_all(SEED) is not required    
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except :
        pass 
    
    try :
        tf.set_random_seed(1234)
    except :
        pass 
declare_reproducible()