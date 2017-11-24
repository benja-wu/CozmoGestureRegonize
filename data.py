import os
import pandas as pd
import numpy as np
import random
from keras.utils import to_categorical
from keras.preprocessing.image import img_to_array, load_img


class DataLoader():
    """
    Base class for loading video data in different formats
    """
    def __init__(self, data_dir, n_videos={'train': None, 'validation': None},
                 labels=None):
        """
        Parameters:
        data_dir: directory containing all data
        n_videos: dictionary specifying how many videos you want to use for both
                  train and validation data; loads all videos by default
        labels  : If None, uses all labels. If a list with label names is
                  passed, uses only those specified
        """
        self.data_dir = data_dir
        self.video_dir = data_dir #os.path.join(self.data_dir, 'videos')
        self.n_videos = n_videos

        self.get_labels(labels)

        self.train_df = self.load_video_labels('train')

        self.validation_df = self.load_video_labels('validation')

        #self.load_validation_labels()

    def get_labels(self, labels):
        path = os.path.join(self.data_dir, 'labels.csv')
        self.labels_df = pd.read_csv(path, names=['label'])
        if labels:
            self.labels_df = self.labels_df[self.labels_df.label.isin(labels)]
        self.labels = [str(label[0]) for label in self.labels_df.values]
        self.n_labels = len(self.labels)

        self.label_to_int = dict(zip(self.labels, range(self.n_labels)))
        self.int_to_label = dict(enumerate(self.labels))

    def load_video_labels(self, data_subset):
        path = os.path.join(self.data_dir, '{}.csv'.format(data_subset))
        df = pd.read_csv(path, sep=';', names=['video_id', 'label'])
        df = df[df.label.isin(self.labels)]

        if self.n_videos[data_subset]:
            df = self.reduce_labels(df, self.n_videos[data_subset])

        #print(df)
        #random.shuffle(df)

        return df

    @staticmethod
    def reduce_labels(df, n_videos):
        """Reduces the amount of videos in a DataFrame to n_videos while
        preserving label distribution"""
        grouped = df.groupby('label').count()
        counts = [c[0] for c in grouped.values]
        labels = list(grouped.index)

        # Preserves label distribution
        total_count = sum(counts)
        reduced_counts = [int(count / (total_count / n_videos))
                               for count in counts]

        # Builds a new DataFrame with no more than 'n_videos' rows
        reduced_df = pd.DataFrame()
        for cla, cnt in (zip(labels, reduced_counts)):
            label_df = df[df.label == cla]
            sample = label_df.sort_values('video_id')[:cnt]
            reduced_df = reduced_df.append(sample)

        return reduced_df


class CNN3DDataLoader(DataLoader):
    """
    Class for loading data to feed into a 3DConvNet
    """
    def __init__(self, data_dir, seq_length, n_videos, labels):
        DataLoader.__init__(self, data_dir, n_videos, labels)
        self.n_videos = n_videos
        self.seq_length = seq_length

    def sequence_generator(self, split, batch_size, image_size):
        """
        Returns a generator that generates sequences in batches
            'image_size' (tuple): Height and width that the images will be
                                  resized to
        """
        if split == 'train':
            df = self.train_df
        if split == 'validation':
            df = self.validation_df

        while True:
            # Load a random batch of video IDs and the corresponding labels
            video_ids, labels = self.random_sample(df, batch_size)
            #print(video_ids)
            #Convert labels to one-hot array
            label_ids = [self.label_to_int[label] for label in labels]
            y = to_categorical(label_ids, self.n_labels)

            # Load sequences
            x = []
            for video_id in video_ids:
                path = os.path.join(self.video_dir, video_id)
                sequence = self.build_sequence(path, image_size)
                x.append(sequence)

            yield np.array(x), np.array(y)

    def build_sequence(self, path, image_size):
        """Returns a 4D numpy array: (frame, height, width, channel)
            'path': Directory that contains the video frames"""
        frame_files = os.listdir(path)
        # add sorted, so we can recognize the currect sequence
        frame_files = sorted(frame_files)
        #print(frame_files)
        sequence = []

        # Adjust length of sequence to match 'self.seq_length'
        frame_files = self.adjust_sequence_length(frame_files)

        frame_paths = [os.path.join(path, f) for f in frame_files]
        for frame_path in frame_paths:
            # Load image into numpy array and preprocess it
            image = load_img(frame_path, target_size=image_size)
            image_array = img_to_array(image)
            image_array = self.preprocess_image(image_array)

            sequence.append(image_array)

        return np.array(sequence)

    def adjust_sequence_length(self, frame_files):
        """Adjusts a list of files pointing to video frames to shorten/lengthen
        them to the wanted sequence length (self.seq_length)"""
        frame_diff = len(frame_files) - self.seq_length

        if frame_diff == 0:
            # No adjusting needed
            return frame_files
        elif frame_diff > 0:
            # Cuts off first few frames to shorten the video
            return frame_files[frame_diff:]
        else:
            # Repeats the first frame to lengthen video
            return frame_files[:1] * abs(frame_diff) + frame_files

    @staticmethod
    def random_sample(df, batch_size):
        """Returns a random batch of size 'batch_size' of video_ids and labels
        """
        sample = df.sample(n=batch_size)
        #print(sample)
        video_ids = list(sample.video_id.values.astype(str))
        labels = list(sample.label.values)

        return video_ids, labels

    @staticmethod
    def preprocess_image(image_array):
        return (image_array / 255. )

labels = ['Swiping Left',
          'Swiping Right',
          'Swiping Down',
          'Swiping Up']
#data = DataLoader(r'C:\Users\Lorenz\Documents\Coding\data\jester_hand_gestures', labels=labels)
#data = CNN3DDataLoader(r'/Users/benja/code/jester-classification-master/data',labels=labels, seq_length =40, n_videos= {'train': 1000, 'validation': 150})
i = 0 
#datasub = data.sequence_generator(split = 'train', image_size= (100, 176) ,batch_size = 10 )

#for i in range (1, 50):
    #print (next(datasub))
