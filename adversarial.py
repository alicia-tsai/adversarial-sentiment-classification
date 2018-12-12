import numpy as np
import torch
import torch.nn.functional as F

from data_loader import DataLoader


# ======================
# Utilities functions
# ======================

def get_input(data_loader, k=0):
    example = data_loader.large_valid.examples[k].text
    label = (data_loader.large_valid.examples[k].label[0] == 'pos') * 1
    word_indices = np.array([data_loader.TEXT.vocab.stoi[word] for word in example])
    one_input = torch.from_numpy(word_indices)

    return one_input.unsqueeze(1), label


def get_logit(input_example, model):
    return model(input_example)


def get_predict(logit):
    return torch.round(torch.sigmoid(logit))


# ===========================
# White box: global search
# ===========================

class GlobalSearchAdversary:
    """White box adversarial attack using global search method."""

    def __init__(self, data_loader=None):
        if not data_loader:
            self.data_loader = DataLoader()
            self.data_loader.load_data()
        else:
            self.data_loader = data_loader

    def custom_loss(self, new_logit, old_logit, new_word_vecs=None, initial_word_vecs=None, data_grad=torch.Tensor([0])):
        loss = - F.mse_loss(new_logit, old_logit) + torch.sum(data_grad ** 2)
        if new_word_vecs is not None and initial_word_vecs is not None:
            # loss -= np.sum(list(map(cosine, new_word_vecs, initial_word_vecs)))
            loss -= torch.sum((new_word_vecs - initial_word_vecs) ** 2)

        return loss

    def attack(self, input_example, model, epsilon=1, similarity_reg=False, perturb_reg=False, print_msg=False):
        # input_example: 2D, tensor([1, number of words])
        if print_msg: print('--- Initial ---')
        initial_logit = get_logit(input_example, model)
        initial_label = get_predict(initial_logit)
        new_logit = initial_logit.clone()

        # initial loss and backpropagation
        loss = self.custom_loss(new_logit, initial_logit)
        model.zero_grad()
        loss.backward(retain_graph=True)
        if print_msg: print('initial loss:', loss)

        success = False
        words_history = [input_example.squeeze(0).clone()]  # all the previous words that have been replaced

        if print_msg: print('\n--- Attack ---')
        while not success:
            # get gradient and compute new embedding
            data_grad = model.embedding.weight.grad[input_example.squeeze(0)].clone()
            input_embedding = model.embedding.weight.data[input_example.squeeze(0)].clone()
            perturbed_embedding = input_embedding - epsilon * data_grad

            new_words_idx = []
            for i, one_embedding in enumerate(perturbed_embedding):
                embedding_distance = torch.sum((one_embedding - model.embedding.weight.data) ** 2, dim=1)
                # set original word and all previously selected word embedding distance to the maximum
                for h in range(len(words_history)):
                    embedding_distance[words_history[h][i]] = float('inf')

                min_idx = torch.argmin(embedding_distance)
                new_words_idx.append(min_idx)

            new_words_idx = torch.from_numpy(np.array(new_words_idx, dtype=int))  # 1D, tensor([number of words])

            # compute new logit and check if attack successfully
            new_logit = get_logit(new_words_idx.unsqueeze(0), model)
            new_label = get_predict(new_logit)

            # compute loss
            if perturb_reg and similarity_reg:
                loss = self.custom_loss(new_logit, initial_logit, perturbed_embedding, input_embedding, data_grad)
            elif similarity_reg:
                loss = self.custom_loss(new_logit, initial_logit, perturbed_embedding, input_embedding)
            else:
                loss = self.custom_loss(new_logit, initial_logit)

            model.zero_grad()
            loss.backward(retain_graph=True)
            if print_msg: print('loss:', loss, '\n')
            words_history.append(new_words_idx.clone())

            if new_label != initial_label:
                break

        return words_history, data_grad, new_logit

    def generate_adversarial(self, model, original_input, words_history, data_grad, replace_ratio=0.1,
                             dist_threshod=50, print_msg=False):
        initial_logit = get_logit(original_input, model)
        initial_label = get_predict(initial_logit)

        # compute the magnitude of the perturb and change from the largest
        grad_magnitude = torch.sqrt(torch.sum(torch.abs(data_grad), dim=1))
        position_to_change = reversed(np.argsort(grad_magnitude))
        total_words = len(position_to_change)

        success = False

        # changing words from the largest perturb
        if print_msg: print('--- Generate Adversary ---')
        new_input = original_input.squeeze(0).clone()
        old_words, new_words = [], []

        for i in range(1, total_words):
            position = position_to_change[i]
            for h in range(1, len(words_history)):
                new_words_idx = words_history[-h]
                original_embedding = model.embedding.weight[original_input.squeeze(0)[position]]
                new_embedding = model.embedding.weight[position]
                distance = torch.sum((original_embedding - new_embedding) ** 2)
                if distance < dist_threshod:
                    break

            # replace word
            new_input[position] = new_words_idx[position]

            if print_msg:
                # clean up html tag
                html_tag = ['<br />', '< br />', '<br', '/>']
                old = self.data_loader.TEXT.vocab.itos[original_input.squeeze(0)[position]]
                new = self.data_loader.TEXT.vocab.itos[new_words_idx[position]]
                for tag in html_tag:
                    old = old.replace(tag, '')
                    new = new.replace(tag, '')
                old_words.append(old)
                new_words.append(new)
                print('\nold words:', old_words)
                print('new words:', new_words)

            new_logit = get_logit(new_input.unsqueeze(0), model)
            new_label = get_predict(new_logit)
            if new_label != initial_label:
                success = True
                break

            # change too many words
            if (i / total_words) > replace_ratio:
                break

        return success, new_input

    def generate_sentence(self, words_idx):
        html_tag = ['<br />', '< br />', '<br', '/>']
        sentence = ' '.join(self.data_loader.TEXT.vocab.itos[id] for id in words_idx)
        for tag in html_tag:
            sentence = sentence.replace(tag, '')

        return sentence
