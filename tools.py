import pickle 

def save(filename, item):
    file_object = open((str(filename)), 'wb')    # Create file object with filename
#    pickle.dump(item, file_object, protocol=4)    # Pickle item
    pickle.dump(item, file_object)    # Pickle item
    file_object.close() 


def read(filename):
    return pickle.load(open(filename, "rb"), encoding='bytes')
