import numpy as np
from xxhash import xxh3_64 as xhash
import itertools
import os
import pickle


def calculate_hash(name, args, kwargs):
    hasher = xhash()
    hasher.update(name)
    hasher.update(str(kwargs.keys()))
    for argument in itertools.chain(args, kwargs.values()):
        hasher.update(str(argument))
    hash = hasher.intdigest()
    print(f'calculated hash :{hash}')
    return hash


# cashe decorator for numpy array or arrays
def numpyDisksave(function):
    def saverfornumpyresults(*args, **kwargs):
        func_name = str(function).split()[2]
        identifier = calculate_hash(func_name, args, kwargs)

        if not os.path.exists(f'./Hcashe/{func_name}/'):
                os.makedirs(f'./Hcashe/{func_name}/', exist_ok=True)

        filepath = f'./Hcashe/{func_name}/{identifier}'
        if os.path.exists(f'{filepath}.npy'):
            return np.load(f'{filepath}.npy')
        elif os.path.exists(f'{filepath}.npz'):
            results = np.load(f'{filepath}.npz')
            return [results[i] for i in results.files]
        else:
            a = function(*args, **kwargs)
            if type(a) is tuple:
                with open(f'{filepath}.npz', 'w+b') as f:
                    np.savez_compressed(f, *a)
            else:
                with open(f'{filepath}.npy', 'w+b') as f:
                    np.save(f, a)
        return a

    return saverfornumpyresults


# cashe decorator for objects
def hCasheobj(function):
    def saverPickleObject(*args, **kwargs):

        func_name = str(function).split()[2]
        identifier = calculate_hash(func_name, args, kwargs)

        if not os.path.exists(f'./Hcashe/{func_name}/'):
                os.makedirs(f'./Hcashe/{func_name}/', exist_ok=True)

        filepath = f'./Hcashe/{func_name}/{identifier}.pkl'
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                result = pickle.load(f)
            return result

        result = function(*args, **kwargs)
        with open(filepath, 'wb') as f:
            pickle.dump(result, f)
        return result

    return saverPickleObject


# cashe decorator with name for objects
def namedhCasheobj(data_name, parameters):
    def hCasheobj(function):
        def saverPickleObject(*args, **kwargs):

            func_name = str(function).split()[2]
            identifier = calculate_hash(func_name, args, kwargs)
            identifier = str(identifier) + str(data_name) + str(parameters)
            if not os.path.exists(f'./Hcashe/{func_name}/'):
                    os.makedirs(f'./Hcashe/{func_name}/', exist_ok=True)

            filepath = f'./Hcashe/{func_name}/{identifier}.pkl'
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    result = pickle.load(f)
                return result

            result = function(*args, **kwargs)
            with open(filepath, 'wb') as f:
                pickle.dump(result, f)
            return result

        return saverPickleObject

    return hCasheobj


# cashe decorator for every element
def instansheCashe(data_name, parameters, syncfrequency=100, args_hashed=1):
    path = f'./Hcashe/{data_name}/{parameters}.pkl'

    if not os.path.exists(path):
        if not os.path.exists(f'./Hcashe/{data_name}/'):
            os.makedirs(f'./Hcashe/{data_name}/', exist_ok=True)
        with open(path, 'w+b') as f:
            new_empty_dict = dict()
            pickle.dump(new_empty_dict, f)

    with open(path, 'r+b') as f:
        cashe: dict = pickle.load(f)

    def period_sync(dictionary):
        i = 0
        while True:
            i = i + 1
            if i % syncfrequency == 0:
                with open(path, 'w+b') as f:
                    pickle.dump(dictionary, f)
            yield i

    sync = period_sync(cashe)
    hasher = xhash()

    def instanshedecorator(function):
        def saverPickleObject(*args, **kwargs):
            hasher.reset()
            for i in range(args_hashed):
                hasher.update(str(args[i]))
            querry = hasher.intdigest()
            result = cashe.get(querry)
            if result is not None:
                return result
            else:
                result = function(*args, **kwargs)
                cashe[querry] = result
                next(sync)
                return result

        return saverPickleObject
    return instanshedecorator
