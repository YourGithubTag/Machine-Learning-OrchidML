# Orchid<sup>ML</sup>


Orchid<sup>ML</sup> is a program to train and test Neural Networks on Flower Recognition designed by Group 31.

### Getting Started:
Simply run the Main python file, the program interfaces using the Command line, so command line is best

The program will display the OrchidML text and then a list of options for models comes up, you can now start setting up jobs for the program

Pick a model by typing the designated letter and pressing enter

Then choose using whether to either start training a new model, or to test an existing Model in the same way<br>
***Warning: If you test before training a new model, there will be no PT file for program to test, and thus the program exits***

Then choose the number epochs you would like to run the model at by typing in a number and hitting enter.

Here you are given the option to run another model right after the one that you have just set up, you can input y and enter, which will then take you through the same
menus to create another model to either run or test an existing one.

Once you have set up all the jobs, at the "set another model" option, type n and hit enter to have all the jobs begin to execute sequentially.
The program will access and download all the neccesary dataset files from the internet and begin to execute all the jobs in the queue.

### Outputs:
All graphs and confusion matrix plots for the training, testing and validation parts of the models are outputted into the same folder that the program is running in.
The results of the training is also outtputed to a CSV file named ***"model name*** Results.csv", which can be used for analysis
The PT file, which holds the model memory, allowing for testing and continued training(see Advanced features) of the model is also outputted into the folder containg the main file
The Precision and F-1 scores are outputted to the command-line.


### Advanced Features

The OrchidML program has advanced features, such as setting the training batches , testing batches, and optimizers for each model and also continuing to train the model from a 
saved state. However you must have an understanding of these features before executing them, as we did not intend for the end user to have this ability, however it is useful to have

To unlock this, simply open up the main file inside your Python editor, and under the job setup function uncomment the marked options inside
the function, this will allow you to access this functionality. 

***However this functionality has been commented out for a reason so use at your own risk.***

Out of Memory errors can be common if the wrong sizes for batches are given and results can be far less accurate if the wrong optimizer is used. 
Running the continue training feature without having a PT file inside the containing folder will also lead to an error.

NOTE: PT Files can be downloaded at this link:
https://drive.google.com/drive/folders/1uZdCUPQlGq4MdXXKovhOU8vPD63Sl3Vh?usp=sharing