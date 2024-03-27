import torch


def ppo(pi_theta, pi_old, ind_mat, eps, reward, device):
    '''
    function for ppo algorithm
    
    input
    ---------------
    pi_theta: logprob of current model
    pi_old: old logprob value
    ind_mat: indicator matrix, 
    eps: epsilon value for clipping
    reward: bleu score
    '''
    # match the shape of old logprobs to current logprobs
    for idx, (theta_logp, old_logp) in enumerate(zip(pi_theta, pi_old)):
        if len(theta_logp) < len(old_logp): # trim old logprob
            pi_old[idx] = old_logp[:len(theta_logp)]
        
        elif len(theta_logp) > len(old_logp): # make old logprob longer
            new_old_logp = torch.zeros_like(theta_logp, dtype=torch.float)
            new_old_logp[:len(old_logp)] = old_logp
            pi_old[idx] = new_old_logp

    pi_old = torch.stack(pi_old).cuda(device)

    ### find the ratio (pi_theta / pi_old)
    ratio = (pi_theta - pi_old) * ind_mat

    ### find surrogate loss
    surr1 = ratio * reward
    surr2 = torch.clamp(ratio, 1-eps, 1+eps) * reward
    
    ### final loss
    RL_term = torch.min(surr1, surr2)

    return RL_term


if __name__ == "__main__":
    pass