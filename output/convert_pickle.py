import pickle


with open('output_train_set.pickle', 'rb') as f:
    train_set = pickle.load(f)
with open('output_test_set.pickle', 'rb') as f:
    test_set = pickle.load(f)

pickle.dump(train_set, open("output_train_set.pickle.v2","wb"), protocol=2)
pickle.dump(train_set, open("output_test_set.pickle.v2","wb"), protocol=2)
