from __future__ import annotations
import pickle
import os
import hashlib
import multiprocessing.pool as mpp

cache_dir = 'cache'

def cached(class_func: bool = False, folder: str = ""):
    if not os.path.isdir(cache_dir): 
        os.mkdir(cache_dir)

    def dec(func):

        def cached_func(*args, **kwargs):
            if class_func:
                args_key = [args[0].__class__.__name__] + list(args[0].__dict__.values()) + list(args[1:] if len(args) > 1 else [])
            else:
                args_key = args
            item_id = str(hashlib.sha224(pickle.dumps(args_key)).hexdigest()) + str(hashlib.sha224(pickle.dumps(kwargs)).hexdigest())


            item_id = item_id.encode('utf-8')
            item_name = str(hashlib.sha224(item_id).hexdigest())
            cache_file = os.path.join(os.path.join(cache_dir, folder), item_name)
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            else:
                if not os.path.isdir(os.path.join(cache_dir, folder)):
                    os.mkdir(os.path.join(cache_dir, folder))
                result = func(*args, **kwargs)
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
                return result

        return cached_func

    return dec


def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError(
            "Chunksize must be 1+, not {0:n}".format(
                chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job,
                                          mpp.starmapstar,
                                          task_batches),
            result._set_length
        ))
    return (item for chunk in result for item in chunk)


mpp.Pool.istarmap = istarmap