import numpy as np
#somewht amazingly, THIS IS IT!? This is ALL you need for a fully fledged minimal HMC implementation. Very simple. Nice.
def hamiltonian_MCMC(n_samples, dVdq,initial_position, path_length=1, step_size=0.5):
    samples = [initial_position]
    momenta = np.random.normal(0,1,[n_samples, initial_position.shape[0]])
    for i in range(n_samples):
        q_new, p_new = leapfrog_integrator(samples[-1],momenta[i,:],dVdq,path_len=path_len,step_size=step_size)
        start_log_p = negative_log_prob(samples[-1]) - np.sum(logpdf(momenta[i,:]))
        new_log_p = negative_log_prob(q_new) - np.sum(logpdf(p_new))
        if np.log(np.random.rand()) < start_log_p - new_log_p:
            samples.append(q_new)
        else:
            samples.append(np.copy(samples[-1]))
    return np.array(samples([1:]))


# theis is the key is figuring out the leapfrog integrator
def leapfrog_integrator(q,p, dVdq, path_len, step_size):
    q,p = np.copy(q), np.copy(p)
    p -= step_size * dVdq(q) / 2 #so the gradient DOES change at every step. This is important to note.
    for _ in range(int(path_len / step_size) -1):
        q += step_size * p
        p -= step_size * dVdq(q)
    q += step_size * p
    p -= step_size * dVdq(q) / 2
    return q, -p
