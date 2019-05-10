import numpy as np
import nltk
from nltk.corpus import wordnet
import gensim
from sklearn.metrics import r2_score
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features=10, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=4)

    def forward(self, input):
        output = self.fc1(input)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)

        return output

class link_evaluation:
    def __init__(self):
        self.net = torch.load('url_relevance.pth')
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

    def get_feature(self, links, topic, relevant_urls):

        # Calculate the edit distance between candidate url and relevant url
        def edit_distance(word1, word2):
            l1, l2 = len(word1)+1, len(word2)+1
            dp = [[0 for _ in range(l2)] for _ in range(l1)]
            for i in range(l1):
                dp[i][0] = i
            for j in range(l2):
                dp[0][j] = j
            for i in range(1, l1):
                for j in range(1, l2):
                    dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+(word1[i-1]!=word2[j-1]))
            return dp[-1][-1]

        # Parse url to n-gram
        def parse_gram(url, n):
            res = set()
            for i in range(len(url) - n + 1):
                res.add(url[i:i + n])
            return res

        url_feature = {}

        # Preprocess to remove http/https/www/non-letter
        for i in range(len(relevant_urls)):
            temp = relevant_urls[i][0].replace("http://","").replace("https://","").replace("www.","")
            relevant_urls[i] = (''.join(filter(str.isalpha, temp)), relevant_urls[i][1])

        for _url, _text in links.items():
            # Preprocess url to remove to remove http/https/www/non-letter
            url = ''.join(filter(str.isalpha, _url.replace("http://","").replace("https://","").replace("www.","")))
            text = _text.split(' ')

            # Edit Distance, tf, 2,3,4,5,6,7-gram apperance rate, Word similarity, Word synonym
            feature = np.zeros(10)

            if len(relevant_urls) > 0:
                # Average edit distance of all relevent url
                for relevant_url, relevant_score in relevant_urls:
                    feature[0] += (len(url) - edit_distance(url, relevant_url)) * relevant_score
                feature[0] = feature[0] / len(relevant_urls)

                # 2,3,4,5,6,7-gram apperance rate
                for ngram in range(2, 8):
                    url_parse = parse_gram(url, ngram)
                    for relevant_url, relevant_score in relevant_urls:
                        relevant_url_parse = parse_gram(relevant_url, ngram)
                        feature[ngram] += len(url_parse & relevant_url_parse) * relevant_score
                    feature[ngram] = feature[ngram] / len(relevant_urls)


            for word in text:
                # Tf
                if word in topic or topic in word:
                     feature[1] += 1

                # Word similarity and synonym
                try:
                    feature[-2] += self.processor.model.similarity(topic, word)
                except Exception:
                    feature[-2] += 0
                try:
                    w1 = wordnet.synset(topic + '.n.01')
                    w2 = wordnet.synset(word + '.n.01')
                    feature[-1] += w1.wup_similarity(w2)
                except Exception:
                    feature[-1] += 0

            feature[1] = feature[1] / len(text)
            feature[-2] = feature[-2] / len(text)
            feature[-1] = feature[-1] / len(text)

            url_feature[_url] = feature

        return url_feature

    def generate_train_test(self, links, topic, relevant_urls, labels):
        size = (int)(len(labels) / 5 * 4)
        features = self.get_feature(links, topic, relevant_urls)
        self.x_train = list(features.values())[0:size]
        self.y_train = label[0:size]
        self.x_test = list(features.values())[size:]
        self.y_test = label[size:]

    def train(self, epochs = 1000, learning_rate = 1e-3):
        train_data = [(torch.FloatTensor(self.x_train[i]),  self.y_train[i]) for i in range(0, len(self.y_train))]
        trainloader = DataLoader(dataset=train_data, batch_size=1, shuffle=True)

        self.net = CNN()

        # Loss function
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.net.parameters(), lr=learning_rate)

        # Train CNN net and loop over the dataset multiple times
        for epoch in range(epochs):

            loss_sum = 0

            for i, data in enumerate(trainloader, 0):
                # Get the inputs
                inputs, labels = data

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward + Backward + Optimize
                outputs = self.net(inputs)

                loss = criterion(outputs, labels)
                loss_sum += loss.item()
                loss.backward()
                optimizer.step()

            if epoch % 50 == 49:
                print("Epoch:", epoch + 1, i + 1, "Loss:", loss_sum)
                loss_sum = 0

        torch.save(self.net, 'url_relevance.pth')

    def test(self):
        test_data = [(torch.FloatTensor(self.x_test[i]),  self.y_test[i]) for i in range(0, len(self.y_test))]
        testloader = DataLoader(dataset=test_data, batch_size=1, shuffle=True)

        self.net = torch.load('url_relevance.pth')

        # Test using trained CNN net
        true_score = []
        pred_score = []
        for data in testloader:
            inputs, outputs = data
            pred_score.append(torch.max(self.net(inputs).data, 1)[1])
            true_score.append(outputs.item())
        pred_score = np.array(pred_score)
        true_score = np.array(true_score)

        print("The test accuracy is:", np.where((true_score == pred_score) == True)[0].shape[0] / len(self.y_test))

    def get_link_score(self, links, topic, relevant_urls):
        features = self.get_feature(links, topic, relevant_urls)
        data_set = [torch.FloatTensor(list(features.values())[i]) for i in range(0, len(features))]
        loader = DataLoader(dataset=data_set, batch_size=1, shuffle=False)
        prediction = []
        for data in loader:
            prediction.append(torch.max(self.net(data).data, 1)[1].item())
        for i in range(len(prediction)):
            features[list(features.keys())[i]] = prediction[i] / 3

        return features
