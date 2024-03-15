import torch
import transformers

from transformers import AutoModel, AutoTokenizer

def ppo(model: AutoModel, ):
    '''
    function for ppo algorithm
    
    input
    ---------------
    model: Language model and also agent
    '''
    ent = torch.log()
    c_ent = torch.log()
    kl_div = ent-c_ent
    pass

def main():
    pass

if __name__ == "__main__":
    main()