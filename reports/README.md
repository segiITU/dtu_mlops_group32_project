# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

Where you instead should add your answers. Any other changes may have unwanted consequences when your report is
auto-generated at the end of the course. For questions where you are asked to include images, start by adding the image
to the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` script that provides two utility functions:

Running:

```bash
python report.py html
```

Will generate a `.html` page of your report. After the deadline for answering this template, we will auto-scrape
everything in this `reports` folder and then use this utility to generate a `.html` page that will be your serve
as your final hand-in.

Running

```bash
python report.py check
```

Will check your answers in this template against the constraints listed for each question e.g. is your answer too
short, too long, or have you included an image when asked. For both functions to work you mustn't rename anything.
The script has two dependencies that can be installed with

```bash
pip install typer markdown
```

## Overall project checklist

The checklist is *exhaustive* which means that it includes everything that you could do on the project included in the
curriculum in this course. Therefore, we do not expect at all that you have checked all boxes at the end of the project.
The parenthesis at the end indicates what module the bullet point is related to. Please be honest in your answers, we
will check the repositories and the code to verify your answers.

### Week 1

* [x] Create a git repository (M5)
* [x] Make sure that all team members have write access to the GitHub repository (M5)
* [x] Create a dedicated environment for you project to keep track of your packages (M2)
* [x] Create the initial file structure using cookiecutter with an appropriate template (M6)
* [x] Fill out the `data.py` file such that it downloads whatever data you need and preprocesses it (if necessary) (M6)
* [x] Add a model to `model.py` and a training procedure to `train.py` and get that running (M6)
* [x] Remember to fill out the `requirements.txt` and `requirements_dev.txt` file with whatever dependencies that you
    are using (M2+M6)
* [x] Remember to comply with good coding practices (`pep8`) while doing the project (M7)
* [x] Do a bit of code typing and remember to document essential parts of your code (M7)
* [x] Setup version control for your data or part of your data (M8)
* [x] Add command line interfaces and project commands to your code where it makes sense (M9)
* [x] Construct one or multiple docker files for your code (M10)
* [x] Build the docker files locally and make sure they work as intended (M10)
* [x] Write one or multiple configurations files for your experiments (M11)
* [ ] Used Hydra to load the configurations and manage your hyperparameters (M11)
* [ ] Use profiling to optimize your code (M12)
* [x] Use logging to log important events in your code (M14)
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code (M14)
* [x] Consider running a hyperparameter optimization sweep (M14)
* [x] Use PyTorch-lightning (if applicable) to reduce the amount of boilerplate in your code (M15)

### Week 2

* [x] Write unit tests related to the data part of your code (M16)
* [x] Write unit tests related to model construction and or model training (M16)
* [x] Calculate the code coverage (M16)
* [x] Get some continuous integration running on the GitHub repository (M17)
* [x] Add caching and multi-os/python/pytorch testing to your continuous integration (M17)
* [ ] Add a linting step to your continuous integration (M17)
* [ ] Add pre-commit hooks to your version control setup (M18)
* [ ] Add a continues workflow that triggers when data changes (M19)
* [ ] Add a continues workflow that triggers when changes to the model registry is made (M19)
* [x] Create a data storage in GCP Bucket for your data and link this with your data version control setup (M21)
* [x] Create a trigger workflow for automatically building your docker images (M21)
* [ ] Get your model training in GCP using either the Engine or Vertex AI (M21)
* [x]  Create a FastAPI application that can do inference using your model (M22)
* [x]  Deploy your model in GCP using either Functions or Run as the backend (M23)
* [ ] Write API tests for your application and setup continues integration for these (M24)
* [ ] Load test your application (M24)
* [ ] Create a more specialized ML-deployment API using either ONNX or BentoML, or both (M25)
* [x] Create a frontend for your API (M26)

### Week 3

* [ ] Check how robust your model is towards data drifting (M27)
* [ ] Deploy to the cloud a drift detection API (M27)
* [ ] Instrument your API with a couple of system metrics (M28)
* [ ] Setup cloud monitoring of your instrumented application (M28)
* [ ] Create one or more alert systems in GCP to alert you if your app is not behaving correctly (M28)
* [ ] If applicable, optimize the performance of your data loading using distributed data loading (M29)
* [ ] If applicable, optimize the performance of your training pipeline by using distributed training (M30)
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed (M31)

### Extra

* [ ] Write some documentation for your application (M32)
* [ ] Publish the documentation to GitHub Pages (M32)
* [x] Revisit your initial project description. Did the project turn out as you wanted?
* [x] Create an architectural diagram over your MLOps pipeline
* [x] Make sure all group members have an understanding about all parts of the project
* [x] Uploaded all your code to GitHub

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

32

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

s241047, s243805, s250394, s250678

### Question 3
> **A requirement to the project is that you include a third-party package not covered in the course. What framework**
> **did you choose to work with and did it help you complete the project?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

We used the transformers library from Huggingface. First we experimented with [T5-base](https://huggingface.co/google-t5/t5-base) for conditional generation, but due to its high computational requirements we switched to [BART-base](https://huggingface.co/facebook/bart-base). BART is an encoder-decoder model, which makes it well suited for the summarization.
We utilized Pytorch Lightning framework for the training step. For our dataset, we picked [pubmed summarization](https://huggingface.co/datasets/ccdv/pubmed-summarization) to explore whether the model could adapt to a different domain while fine-tuning it for summarization.

## Coding environment

> In the following section we are interested in learning more about you local development environment. This includes
> how you managed dependencies, the structure of your code and how you managed code quality.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Recommended answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development environment, one would have to run the following commands*
>
> Answer:

We used conda, pip, and git to manage our dependencies. To get out dependencies a new member would need to clone this repository, create a conda environment, and then install dependencies in the requirements files. 

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. What did you fill out? Did you deviate from the template in some way?**
>
> Recommended answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
>
> Answer:

We initialized the project with the MLOps-specific cookiecutter template. We have removed the ``notebooks/`` folder as we have not created any notebooks to showcase this project, as well as the ``dockerfiles/`` folder as all our dockerfiles were added to the root directory for ease of access. Other than that, we have have been committed to the project structure which was provided with the template, as we have found that this structure is very good for maintaining a structured MLOps project with its dedicated ``src/``, ``data/``, ``tests/`` and ``models/`` folders being particularly useful. The template scripts and the sub-directories which were also provided in the template have also been very useful, and we have largely used these as a starting point for our project. 

### Question 6

> **Did you implement any rules for code quality and format? What about typing and documentation? Additionally,**
> **explain with your own words why these concepts matters in larger projects.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used ... for linting and ... for formatting. We also used ... for typing and ... for documentation. These*
> *concepts are important in larger projects because ... . For example, typing ...*
>
> Answer:

We did not implement any definitive rules for code quality and styling, as we spent a long time solving technical issues, and thus had less time to complete some of the portions which we deemed more important. This meant that styling and code quality was deprioritized We did use typing in some portions, mainly our data, model and training scripts. Especially in these scripts, it was important for us to have consistency and readability. We also did our best to create meaningful documentation that is neither too vague or too descriptive.
Maintaining good code quality and formatting, as well as proper typing and documentation, is incredibly important in coding porjects on a larger scale, as they provide consistency throughout the project. Additionally they provide greater readability for anyone that are attempting to familiarize themselves with the code. This means that a new member of the project can easily begin creating code that follows the same standards and practices, and thus their code should have the same level of readibility and consistancy as the rest of the project. This means that another new member is again able to quuickly and easily interpret and write code.

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement and what are they testing in your code?**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *In total we have implemented X tests. Primarily we are testing ... and ... as these the most critical parts of our*
> *application but also ... .*
>
> Answer:

We implemented 15 tests in total. These are primarily focused testing the data, our model, but also testing the API were implemented. In testing the data, we made sure it had the correct struture, that the abstract was shorter than the full article, and that the abstract and article had some overlap. We also tested the methods the model used for training to ensure they would generate loss upon stepping forward. In future iterations we would also want to test the output of the model and make sure the abstracts it creates are shorter and has overlap. Lastly, we tested the API by making sure it responds as we expect. 

### Question 8

> **What is the total code coverage (in percentage) of your code? If your code had a code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer:

Our code has a coverage of [86]%, with the distribution visible in the table below.
| Name                                       | Stmts | Miss | Cover |
|--------------------------------------------|-------|------|-------|
| app/__init__.py                            | 0     | 0    | 100%  |
| src/dtu_mlops_group32_project/__init__.py  | 5     | 0    | 100%  |
| src/dtu_mlops_group32_project/data.py      | 37    | 15   | 59%   |
| src/dtu_mlops_group32_project/main.py      | 33    | 18   | 45%   |
| src/dtu_mlops_group32_project/model.py     | 37    | 3    | 92%   |
| tests/__init__.py                          | 5     | 0    | 100%  |
| tests/integrationtests/test_api.py         | 7     | 0    | 100%  |
| tests/unittests/test_data.py               | 31    | 0    | 100%  |
| tests/unittests/test_model.py              | 21    | 1    | 95%   |
|--------------------------------------------|-------|------|-------|
| **TOTAL**                                  | 176   | 37   | 86%   |

Even if our code coverage was at 100%, then we can not trust that it is free from errors and bugs. This is due to the fact that the code coverage simply explains how much of the code is executed when running tests. Thus, the tests could insufficient in detecting all the errors or edge cases, and thus we can not trust that 100% coverage equals error free code.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

We did not utilize pull requests as much as we perhaps should, as they were mainly used to set up GitHub dependencies. We did make use of multiple upstream working branches in addition to the main branch, as a form of branch protection. As each member was working on different items, we did not feel that it was necessary for each member to have their own dedicated working branch. We rarely had merge conflicts, and those were easily resolved. In future projects like this, we would however likely create dedicated branches for each person and maybe each feature to ensure no issues by working on the same branch. This setup did mean we could each participate on the working branches, and then push to the main branch once a part was working properly. The importance of branches and pull requests becomes epsecially clear when many people are working on the same project. With these, you are able to create structured versions of the project, with each pull request being an update of working code, without issues of multiple people working on the same problem.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

We used DVC with Google Drive which made us able to store the data remotely and avoid having the data stored within the pipeline on GitHub. This would have helped immensely with controlling the data, as any potential changes to the data would otherwise mean having to upload the entire dataset and any changes made to it onto GitHub. We made minimal changes to our data after preprocessing it, so the main benefit of using it was to avoid storing the dataset on GitHub.

### Question 11

> **Discuss you continuous integration setup. What kind of continuous integration are you running (unittesting,**
> **linting, etc.)? Do you test multiple operating systems, Python  version etc. Do you make use of caching? Feel free**
> **to insert a link to one of your GitHub actions workflow.**
>
> Recommended answer length: 200-300 words.
>
> Example:
> *We have organized our continuous integration into 3 separate files: one for doing ..., one for running ... testing*
> *and one for running ... . In particular for our ..., we used ... .An example of a triggered workflow can be seen*
> *here: <weblink>*
>
> Answer:

We are running unit testing and integration tests. Our tests run on  three operating systems: Ubuntu, Qindows, and Macos. We test for both python 3.12 and 3.11. We do make use of catching by utilizing pip and pyproject.toml. You can find an example workflow at this [link]( https://github.com/segiITU/dtu_mlops_group32_project/actions/runs/12957332195)

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Recommended answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: Python  my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We did not use config files for our experiments, apart from when we did a sweep of different configurations using ``wandb``. We decided to hardcode the configurations into the training script, but with the ability to change them through the use of a simple argparser. To run an experiment while changing one of the hyperparameters, one would run the following command

``Python train.py --lr 2e-3 --batchsize 10``

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

To ensure that no information is lost, we utilised ``wandb`` to record the runs and the respective results from different hyperparameter configurations. One of the hyperparameters in the training script is a seed, which is used to make the experiments reproducible. This seed ensures that when training a model with a specific set of hyperparameters, the same results can be achieved. This seed can also be changed by parsing a different seed through the argument parser. 

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Recommended answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

As seen in the first figure below, we have tracked the training loss and the validation loss. From this image we see that both the training and validation loss decrease nicely, for some of the runs. Notably, the second sweep does not even start, and sweep 3 encounters an issue which massively increases both the training and validation loss. We do see that for the first, fourth and fifth sweep, that both the training loss and validation loss begins to plateau. It can also be seen that some runs, the Early stopping halted the training process to avoid overfitting. In the second figure below we see the how the parameters impacted the validation loss. It seems we trained the model with the configurations set up incorrectly, as we idealy would train all the hyperparameters separately and not joined like this. If we had more time to train the model using wandb we would do a full sweep of all the parameters to test which combination actually gives the best result. It is difficult to see from this image, but sweep 1 and 4 has almost exactly the same parameters and thus yield very similar results. Sweep 5 has a lower learning rate, but also yields a similar validation loss as sweep 1 and 4. 
![first figure](figures/Wandb_full.png)
![second figure](figures/Wandb_graph.png)

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments/project? Include how you would run your docker images and include a link to one of your docker files.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

For our project, we developed several images: [one "baseline" Docker image](predict.dockerfile) to be able to create a container for inference both locally and on the Google Cloud Platform (GCP), as well as two images - one [front end](frontend.dockerfile) and one [backend image](backend.dockerfile) - to create containers for a User Interface (created with the Gradio package) and a backend container with our fine-tuned model. To test the images and running containers, we first ran them locally with Docker Desktop. Then, we deployed the images to GCP's Artifact Registry, where they could be re-used in different application in Cloud Run. You can view our front-end containerized UI [by clicking here](https://bart-summarizer-frontend-962941447685.europe-west1.run.app) - it will likely only be running a couple of days after exam submission 24-Jan-2025. 

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

When encountering bugs while running experiments, we first manually read the error messages and reviewed our code to identify the source of the issue. If the issue persisted, we would test smalller sections of the code and go through the documentation to isolate the mistake. In addition, we also consulted online resources in case similar bugs with known solutions existed. We did not utilize debugging tools such as the Python debugger, though it would likely have been helpful. 
We did not try profiling our code, not because we think it's already perfect, but because we prioritised other tasks. Profiling the code would have been good, to check that it is not being called more than necessary and that no tasks take longer than expected.

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Recommended answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

We utilsed GCP's: 
* Artifact Registry for storing Docker images to be deployed on different services, e.g. Cloud Run.
* Cloud Run for running the containers created from the images in Artifact Registry.
* Cloud Build for triggering an automatic Cloud Build from local dockerfile --> image in Artifact Registry --> Cloud Run. The trigger was based on a push to main, so we soon had to update the trigger require approval because we pushed so much. 

We also experimented with Vertex AI, i.e. GCP's AI platform, and Compute Engine (GCP's version of virtual machine), but we did not finally implement these in our project. 

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

Unfortunately, we did not fully utilize this component of the cloud. Our problems with getting decent GPUs over the cloud prevented us from tranining the model over cloud. We only experimented with the Vertex AI to see how long would training take on CPUs.
We experimented with using Compute Engine resources for training our model. We experimented with initializing a NVIDIA V100 and NVIDIA Tesla P4 on the europe-west4 cluster. Since we were unfamiliar with Compute Engine before this course, it was quite time-consuming to initialize an VM and even more so to be able to train a model, so we decided to do our training on a HPC, which we are more familiar with. 

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

![image](https://github.com/user-attachments/assets/a3292061-ab3c-4a01-9f5c-2435255d6219)

![image](https://github.com/user-attachments/assets/4f05a492-99a4-4352-8bfc-87e4dd217871)

### Question 20

> **Upload 1-2 images of your GCP artifact registry, such that we can see the different docker images that you have**
> **stored. You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

![image](https://github.com/user-attachments/assets/f86ca9d2-724b-4e7b-9e1a-1d0ddec0cae3)

![image](https://github.com/user-attachments/assets/e574c02e-c38b-4c52-a2ec-04412eb42806)


### Question 21

> **Upload 1-2 images of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:
![image](https://github.com/user-attachments/assets/13131096-6a79-4542-9624-d786bff2cadc)

![image](https://github.com/user-attachments/assets/520443d0-db6e-4b25-ba6f-d3ef9204370c)

### Question 22

> **Did you manage to train your model in the cloud using either the Engine or Vertex AI? If yes, explain how you did**
> **it. If not, describe why.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We managed to train our model in the cloud using the Engine. We did this by ... . The reason we choose the Engine*
> *was because ...*
>
> Answer:

We could not manage to train the model in the cloud. In our region, not many GPU types were available. That is why neither the Compute Engine, nor the Vertex AI were completely implemented, though we did start training runs over Vertex AI to see how slow it was on CPU.

## Deployment

### Question 23

> **Did you manage to write an API for your model? If yes, explain how you did it and if you did anything special. If**
> **not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did manage to write an API for our model. We used FastAPI to do this. We did this by ... . We also added ...*
> *to the API to make it more ...*
>
> Answer:


Yes, I created an API for the BART-based text summarization model using FastAPI. The API has two endpoints: a POST endpoint /summarize for summarizing input text and a root GET endpoint to verify the API is running. The model is loaded from a file (final_model.pt) using PyTorch, and if it's not available locally, it is downloaded from Google Cloud Storage (GCS). Once the model is loaded, the input text is tokenized, and the summary is generated using the BART model's generate function. To ensure the model is ready before requests are processed, an asynchronous context manager loads the model during the app's lifespan. This API is designed to efficiently handle requests by leveraging FastAPI's asynchronous features.

We did manage to write an API for our model using FastAPI - see our [main.py script here](src\dtu_mlops_group32_project\main.py). We created a POST endpoint called /summarize/ to accept input as a text file. We could have made the API more robust, if time permitted. 



### Question 24

> **Did you manage to deploy your API, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

Yes, we did manage to deploy our API, both locally and in the cloud. You can view an example of our deployed API which is publically available via URL https://bart-summarizer-962941447685.europe-west1.run.app. If you want to try to call the API and use our summarization model you can use curl with the following example:

>curl -X POST -F "file=@C:<file_location/>input.txt" https://bart-summarizer-962941447685.europe-west1.run.app/summarize/

### Question 25

> **Did you perform any unit testing and load testing of your API? If yes, explain how you did it and what results for**
> **the load testing did you get. If not, explain how you would do it.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *For unit testing we used ... and for load testing we used ... . The results of the load testing showed that ...*
> *before the service crashed.*
>
> Answer:

To unittest the API, we simply tested whether the API worked and responded with the expected status code.

### Question 26

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

We did not manage to implement any tools for monitoring the data due to time constraints. Though, if we had time to do so, we would have liked to implement system monitoring of our Google cloud service system, to make sure it alerts us if it encounters an issue or if the model is not behaving as expected. The creation of these alert constraints are time-consuming and the reason we omitted implementing them.

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 27

> **How many credits did you end up using during the project and what service was most expensive? In general what do**
> **you think about working in the cloud?**
>
> Recommended answer length: 100-200 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ... . Working in the cloud was ...*
>
> Answer:

Group member s250678 spent all 50 free USD credits during the exercises and activated another free billing account. It is difficult to get a cost breakdown of a billing account when it uses free credits, so it has not been possible to find out exactly what services drained the $50 free credits, but he presumably activated a service and forgot to turn it off again.
s250394 spent 3.07 overall, but it is difficult to assess how much was spent on the exercises and how much was spent on the project.

### Question 28

> **Did you implement anything extra in your project that is not covered by other questions? Maybe you implemented**
> **a frontend for your API, use extra version control features, a drift detection service, a kubernetes cluster etc.**
> **If yes, explain what you did and why.**
>
> Recommended answer length: 0-200 words.
>
> Example:
> *We implemented a frontend for our API. We did this because we wanted to show the user ... . The frontend was*
> *implemented using ...*
>
> Answer:

We only implemented a frontend with Gradio, which you can [see here](https://bart-summarizer-frontend-962941447685.europe-west1.run.app). Unfortunately, we did not manage to link our trained model on the backend with this frontend UI, so you are not able to do a summatization task with the link provided. 

### Question 29

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally, in your own words, explain the**
> **overall steps in figure.**
>
> Recommended answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and push to GitHub, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

![third figure](figures/workflow.png)

We start with the developer node and the immediate connections show that we version controlled our project via github. We also utilized W&B for training, and used HPC to complete it. Afterwards, we used docker containers to capture our model. These Docker containers were integrated into a cloud pipeline, which utilizes Cloud Build Triggers for data control and Artifact Registry to manage and deploy the model for inference. We interfaced with our application through Cloud Run, using the FastAPI framework. Finally, we implemented a user interface to complete the workflow.

### Question 30

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Recommended answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

The biggest challenge we faced was the high computational demand of the t5-model we tried to run. Unfortunately, we were unable to get GPUs in the cloud, this proved that the we had to discard our first core files that we constructed for T5. Given the time restriction of the project, we decided to scale down to a BART-base, and use the HPC of s250394's university for training, as with cloud CPU this would also take several days. We also had to scale down our data. 

### Question 31

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project. Additionally, state if/how you have used generative AI**
> **tools in your project.**
>
> Recommended answer length: 50-300 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
> *We have used ChatGPT to help debug our code. Additionally, we used GitHub Copilot to help write some of our code.*
> Answer:

s250678 was in charge of developing of setting up the initial cookie cutter project and developing of the docker containers for deploying our application. He has used genAI tools to help adjust the dockerfiles and with creating CLI inputs for deploying to the GCP.
s250394 developed the data processing, model and training files. She also run the model training, and was responsible for W&B. She also helped with testing and continious integration. She has used genAI tools for debugging the code.
s243805 is responsible for the creation of the FastAPI and the integration with the rest of the code, as well as for the dockerfile and configuration files for its deployment. He has used genAI to help solve bugs and errors in the code. s241047 was in charge of testing and mainly responsible for the report, but due to technical issue were limited in what they could provide. Has contributed to model and training files as well as general continuous integration, and has generally helped where able. Has used genAI tools for debugging code.

