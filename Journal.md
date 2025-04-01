# First Week - 15/Jan/2025
## Task before next meeting 
- Familiarize yourself with conda and pytorch
- Create conda virtual environment for the project first
- After that, try replicating the Jersey Number Recognition model with the help of the paper and source code
- Try submitting the work on EvalAI

# Second Week- 22/Jan/2025
## Work done

### Tanner 
- Read [Mkoshkina](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Koshkina_A_General_Framework_for_Jersey_Number_Recognition_in_Sports_Video_CVPRW_2024_paper.pdf) as well as [Low resolution Jersey Number Recognition](https://arxiv.org/pdf/2309.06285v1)
- Downloaded and setup SoccerNet tracklets/data
- Working on replication of [Mkoshkina](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Koshkina_A_General_Framework_for_Jersey_Number_Recognition_in_Sports_Video_CVPRW_2024_paper.pdf)
- Created my Venv in conda 
### Henry
- Downloaded and setup the SoccerNet data and framework
- Reading "A General Framework for Jersey Number Recognition in Sports Video" and "Jersey Number Recognition using Keyframe Identification from Low-Resolution
Broadcast Videos "
### Leo
- Set up conda venv
- Setup [Mkoshkina](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Koshkina_A_General_Framework_for_Jersey_Number_Recognition_in_Sports_Video_CVPRW_2024_paper.pdf)
- Downloading SoccerNet data
- Reading [Mkoshkina](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Koshkina_A_General_Framework_for_Jersey_Number_Recognition_in_Sports_Video_CVPRW_2024_paper.pdf) as well as [Low resolution Jersey Number Recognition](https://arxiv.org/pdf/2309.06285v1)
### Antonio 
- Set up conda venv
- Began replicating, but working through errors: [Mkoshkina](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Koshkina_A_General_Framework_for_Jersey_Number_Recognition_in_Sports_Video_CVPRW_2024_paper.pdf)
- Reading supplied readings

## Task before next meeting
- Finish downloading the data and replicating the work
- Read both papers that are provided in supporting_materials.docx
- Start searching for the recent research papers on **scene text recognition**
- Checkout Colab

# Third Week 29/Jan/2025
## Work done
### Tanner 
- Set up environment, Got testing working for soccernet and I believe training as well.
- Might switch to colab
### Henry
- Read two papers on Jersey Number Recognition
- Setup the code for the jersey-number-pipeline
- Testing the inference and training of the model
- Setting up the google colab
### Leo
- Still setting up on local machine
- Reviewed the tools used in the pipeline
### Antonio 
- Still setting up on local machine
- Trying google colab approach, since M1 mac lacks cuda
### Imam
- Setting up the project in Google Colab
- Encountered issues with the pipelines multiple virtual env setup.

# Fourth Week 05/Feb/2025
## Work Done
### Tanner 
- Set environment up completely. Fixed all the dependency issues. Was able to reproduce the results from the [Mkoshkina](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Koshkina_A_General_Framework_for_Jersey_Number_Recognition_in_Sports_Video_CVPRW_2024_paper.pdf) paper.
- Read research paper on Scene text recognition. Trying to understand what we can improve on in this pipeline (Most likely will be pulling a component out and putting a new one in I think)
### Leo 
- Ran model on laptop, memory leak, looking into alternatives for running the project
### Antonio 
- Got model working locally, but ran into memory leak
### Imam
- Replicated the pipeline in remote desktop
- Resolved all dependency issues
  

# Fifth Week 12/Feb/2025
## Work Done
### Antonio 
- Reading research papers
### Leo 
- Reading relevant papers on jersey recognition and finding additional datasets
### Imam
- Looked into the background study
- Potential areas of improvements - adding DeblurGAN-v2 to tackle motion blur
### Tanner
- Looked into potential replacement parts for the pipeline. Looked into scene text recognition for hazy/blurry weather.
- looked into the bench marks for vitpose and parseq
### Mariya
- Read and reviewed [Mkoshkina](https://openaccess.thecvf.com/content/CVPR2024W/CVsports/papers/Koshkina_A_General_Framework_for_Jersey_Number_Recognition_in_Sports_Video_CVPRW_2024_paper.pdf)
- Downloaded soccernet dataset
- Tried setting up project on google collab, but ran into issues.

# Sixth Week 21/Feb/2025
## Work Done 
### Leo 
- Looked into models with better performance, to find which models could be swapped. Read more papers about jersey recognition.
### Tanner 
- Read the required papers discussed from last meeting
- Looking for better models to swap into pipeline. Found 2.
### Mariya
- Looked into newer models that could enhance the pipeline. 
- Assessed the different benchmarks for newer models compared to the existing ones in the pipeline.
- Looked at the output files from Tanners replication to assess the efficiency of the current models in the pipeline, and where improvements can be made.
### Imam
- Looked into other data preprocessing techniques.
- Figured that super resolution is necessary for further improvement.

# Seventh week 26/Feb/2025
## Work Done

### Tanner 
- Working on Project proposal and task deligation
- Looking into RESNET50 for legibility classifier
### Leo 
- Researching superresolution models
### Imam
- Setting up the environment for DeblurGANv2.
- Resolved all dependency issues.

# Eighth Week 5/March/2025 
## Work Done 

### Tanner 
- Setting up more powerful environment for running entire pipeline since my desktop is now getting MEM errors
- Working on implementing RESNET50
### Leo 
- Researching superresolution models, specifically for Scene Text Recognition (STR)
- Working on implementing PERN, a superresolution model for STR
### Imam
- Replicated DeblurGANv2 as mentioned in the paper.
- Ran it on both pretrained backbone.

# Ninth Week 12/March/2025
# Work Done

### Tanner 
- Got environemnt replicated again
- Implemented resnet50 into pipeline
- Got slightly bettern results than SOTA pipeline
### Leo 
- Working on implementing PERN, a superresolution model for STR, running into issues implementing on local machine
- Looking into alternative models
### Imam
- Applied DeBlurGANv2 on train and test dataset.

# Tenth week 19/March/2025 
## Work Done 

### Tanner 
- Fine tuned legibility classifier and resnet50 with weakly labelled datasets
- Fine tuned parseq on weakly labelled dataset
- Results were again slightly better than our previous and SOTA
- Used deblurred challenge dataset on EVAL ai (3% improvement from SOTA)
- Working on training new legibility classifier and fine tune parseq on new deblurred dataset and comparing it to previous result
### Leo 
- Ran Post Legibility Classifier Crops through Real-ESRGAN Superresolution model
### Imam
- Ran DeblurGANv2 on challenge dataset and the legibility classifer image set

# Eleventh Week 25/March/2025

### Imam
- Polishing the project and getting everything wrapped up for the presentation.
- Worked on the presentation slides.

 
