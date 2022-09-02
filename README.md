In this project, I have created a program to classify images of cars and tell if an image is of a car or not.
For the classification of the images, I have used an image classifier known as keras.
Sequential model which is a plain stack of layers where every layer has its own input and output tensor.
After this, the data is loaded using keras.preprocessing.image_dataset_from_directory.

***Importing all the necessary libraries***

This is where I started my code and imported the necessary libraries that I needed to make the image classifier work. 

![image](https://user-images.githubusercontent.com/79290729/188218499-f9e62907-6859-4347-815b-34a5ef4db97b.png)

Here are all the libraries that Ire used in this project


***Creating a dataset***

After the libraries Ire imported, I need to create a training class and then, define a dataset.


![image](https://user-images.githubusercontent.com/79290729/188224711-6912f718-daaf-4e14-a493-87f1ccf47b73.png)


The dataset was defined by first defining some parameters for the loader


![image](https://user-images.githubusercontent.com/79290729/188224732-c9a1428c-d348-4ced-a1f1-cbe09b09c1c5.png)


After this, I split the dataset into two parts. The first part can be used to train the dataset and the second one can be used to validate the model. I have split the dataset in parts of 80% for training and 20% for validation. 


 ![image](https://user-images.githubusercontent.com/79290729/188230464-f0a552a3-2e19-4ee6-8cd6-755967e8eebc.png)

 
And here is the output that I get from this step – 


![image](https://user-images.githubusercontent.com/79290729/188230575-1e53044d-8831-4812-b34a-6bfe729d65bc.png)


***Passing the data over***


I get the data ready to be passed over to model.fit in batches to measure how well the model was differenciating.


![image](https://user-images.githubusercontent.com/79290729/188234848-1848f35e-2b4f-4931-b046-3b638dfba998.png)


After I hand the data over, I now need to make sure that the data can be retrieved from the disk without any problems. This is done by using dataset.cache() which keeps the images in memory after they are retrieved off the disk. All of this takes place during the first epoch and helps preventing bottlenecks in the data flow


![image](https://user-images.githubusercontent.com/79290729/188234926-cb5525e9-a094-485d-9d2a-db3c5445213d.png)


***Compiling the model***

For every training epoch, I have monitored the training and validation by using the optimizers.Adam optimizer and losses.SparseCategoricalCrossentropy loss function. Here is the code for the same – 
 

![image](https://user-images.githubusercontent.com/79290729/188235499-547a74be-2252-460a-86a4-1e6ee43a7bd4.png)


***Model summary***


In This step, I created a new class named, model.summary() to view the layers, to see the summary of the training and then to visualize the results by plotting a graph. Here is how the class is created and the model is trained – 


![image](https://user-images.githubusercontent.com/79290729/188236676-82a29fb7-42ce-4d16-b12b-40f6a2c94e96.png)


And here is how I create the graphs for the summary – 


![image](https://user-images.githubusercontent.com/79290729/188239750-4044e017-c22c-47f3-b663-d7f69c98280c.png)


Here are the results of the graph


![image](https://user-images.githubusercontent.com/79290729/188239864-4d4b772a-2f81-40cf-a892-511bf09bcf85.png)


![image](https://user-images.githubusercontent.com/79290729/188240019-28375f62-ba10-454a-96e5-eef7a9ee4331.png)


***final validation*** 


After the training is done, it is finally time to test the new data and see how the program performs. To test the new data, I created a class called test_data. Here is the code for the same 


![image](https://user-images.githubusercontent.com/79290729/188240230-5ffdf432-3767-475d-84d7-3826ccb5d22b.png)


Here are the results that I got by giving it a sample image


![image](https://user-images.githubusercontent.com/79290729/188240269-5725cc56-62e6-457f-9f92-1dada7c82486.png)
![image](https://user-images.githubusercontent.com/79290729/188240285-9a5c992f-dc13-4a32-87a6-b4c19b7838c9.png)
![image](https://user-images.githubusercontent.com/79290729/188240330-2cd6e493-7c18-485d-8623-0f76c657a5dc.png)

And so I can confirm that the image classifier works!
