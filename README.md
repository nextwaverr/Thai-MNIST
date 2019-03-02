# Thai-MNIST
Thai digit handwriting and example code
   Thai-MNIST is a dataset of thai handwrite digit images—consisting of a training set of 280,000 examples and a test set of 31,000 examples (75,000 examples and a test set of 8,400 examples in Alphabet Class) Each example is a 28x28 grayscale image, associated with a label from 10 classes. We intend Thai-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.
<p align="center">
<img src="image/Thai_MNIST_Number.jpg" width="30%" height="30%">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="image/Thai_MNIST_Alphabet.jpg" width="30%" height="30%">
</p>

# Download Dataset
- Number dataset for train and validate [download](https://drive.google.com/open?id=1cZmfAfYXegdGGNISboq7pvPR-KnXmduw)
- Alphabet dataset for train and validate [download](https://drive.google.com/open?id=1VMIEdmp_uPqywq3Kcd4HLwhfpbbFCHX5)

- Number Image for train and validate [download](https://drive.google.com/file/d/1eWOB4igPMWhWRJza6y5CkpT6pq35rgE2/view?usp=sharing)
- Alphabet Image for train and validate [download](https://drive.google.com/open?id=1Ne7cvbKGq9e5TeUNGKDmGgzz0tsqWAE6)

# Requirement
- Python  >= 3.4
  - TensorFlow >= 1.0
  - keras
  - numpy
  - cv2
  - PIL
  
## Usage
Download and Extract dataset to dataset/number 

if you download alphabet class Extract dataset to dataset/alphabet

training model 
```python
python Thai_MNIST_Example.py
```
then testing model 
```python
python prediction_Thai_MNIST.py path/to/your/imge
```
We have a study section in jupyter notebook .
```python
Thai_MNIST_Example.ipynb
```
We gets to 99.26% test accuracy after 10 epochs (Number class) and 99.58% (Alphabet class)

## Fast way if there are problems or questions
E-mail : chatchai@nextwaver.com

Tel : 0858242000

Line ID : 0858242000

## Contact Me

E-mail : chatchai@nextwaver.com

Tel : 0858242000

Line ID : 0858242000

## Support and donate
E-mail : chatchai@nextwaver.com

Tel : 0858242000


## License
[Creative commons](https://creativecommons.org/licenses/by-nd/4.0/)

## Sponsor
<p align="center">
<img src="image/OSEDA.jpg" width="30%" height="30%">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="image/Where in Thailand.png" width="30%" height="30%">height="30%">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="image/True.png" width="30%" height="30%">
</p>
