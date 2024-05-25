
def reshape_for_vmap(x):
    if hasattr(x,"shape"):
        if  len(x.shape) == 0:
            return x.reshape([1])
        return x.reshape((x.shape[0], 1)+ tuple(x.shape[1:]))
    else:
        return x