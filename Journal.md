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
### Antonio 
- Got model working locally, but ran into memory leak
### Imam
- Replicated the pipeline in remote desktop
- Resolved all dependency issues

# Fifth Week 12/Feb/2025
## Work Done
### Antonio 
- Reading research papers
### Imam
- Looked into the background study
- Potential areas of improvements - adding DeblurGAN-v2 to tackle motion blur
