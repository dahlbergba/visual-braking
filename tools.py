import pickle 

"""This file contains some very short wrappers for the pickle module that can be used to easily 
save and read data, particularly when working with binary files."""

def save(filename, item):
    file_object = open((str(filename)), 'wb')    # Create file object with filename
#    pickle.dump(item, file_object, protocol=4)    # This line is helpful when Python 2 compatibility is needed
    pickle.dump(item, file_object)    # Pickle item
    file_object.close() 

def read(filename):
    return pickle.load(open(filename, "rb"), encoding='bytes')
