
import cv2
import os
import pandas as pd
import os
import cv2
import csv
import pandas as pd
import datetime
import json
import lzma
from sklearn.model_selection import train_test_split


root_dir = "./Dataset/Data"


# -------------------MP4 to image---------------
def convert_mp4_to_image():
    # path to the directory containing the user folders conatining the videos
    
    Users = os.listdir(root_dir)

    # loop over all the users in the directory
    for user_folder in Users:
        path_to_user = root_dir+"/"+user_folder
        
        # loop over files in each directory
        for file_name in os.listdir(path_to_user):
            if file_name.endswith('.mp4'):
                
                # read the video file
                video = cv2.VideoCapture(os.path.join(path_to_user, file_name))
                
                # get the first frame of the video
                success, image = video.read()

                # save the frame as an image file
                if success:
                    image_path = os.path.join(path_to_user, os.path.splitext(file_name)[0] + '.jpg')
                    print(image_path)
                    cv2.imwrite(image_path, image)
                    os.remove(path=os.path.join(path_to_user, file_name))
                else:
                    print(f'Failed to extract first frame from {file_name}')

                # release the video capture object
                video.release()

# Generate the new dataset from files and classification.csv

def extract_publication(file_list,subdir):
  dates=[]
  captions=[]
  medias=[]
  likes=[]
  comments=[]
  for file_name in file_list:
    # Date
    date = datetime.datetime(int(file_name[0:4]),int(file_name[5:7]), int(file_name[8:10]), int(file_name[11:13]), int(file_name[14:16]), int(file_name[17:19]))
    date = date.strftime('%Y-%m-%d %H:%M:%S')
    dates.append(date)

    # Caption
    file_path = os.path.join(subdir, file_name)
    with open(file_path, 'r') as file:
      caption=file.read()
      captions.append(caption)
      
    # Comments and likes
    json_file_path = os.path.join(subdir, file_name)[:-3] + "json.xz"
    
    with lzma.open(json_file_path, 'rb') as f:
        data = json.loads(f.read().decode())
    like= data['node']['edge_media_preview_like']['count']
    comment=data['node']['edge_media_to_comment']['count']
    likes.append(like)
    comments.append(comment)

    # Get images
    media = str(subdir)+'/'+file_name[:-3]+"jpg"
    if ( os.path.exists(media)== False) :
      media = str(subdir)+'/'+file_name[:-4]+"_1.jpg"


    medias.append(media)


  return dates, captions, medias, likes, comments


def generate_dataset_from_data():
    users = []
    dataset=[]

    for subdir, dirs, files in os.walk(root_dir):
        user_name=subdir[43:]
        dates=[]
        captions=[]
        medias=[]
        likes=[]
        comments=[]
        file_list = [file for file in files if file.endswith('.txt')] 
        dates, captions, medias, likes, comments=extract_publication(file_list,subdir)
        if len(medias) != 0 and len(captions) != 0 : 
            dataset.append([user_name,dates, captions, medias, likes, comments])
    print('dataset', dataset)
    
    with open('./Dataset/new_dataset.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['user_name', 'Dates', 'Captions', 'Medias', 'Likes' , 'Comments'])
        for row in dataset:
            writer.writerow(row) 

def merge_datasets():
    # load the new_dataset and the classification_dataset
    new_dataset = pd.read_csv('./dataset/new_dataset.csv')
    classification_dataset = pd.read_csv('./dataset/classification_dataset.csv')

    # Edit usernames ending with ***
    for i in range(0,classification_dataset['user_name'].size):
        classification_dataset['user_name'].iloc[i]
        if classification_dataset['user_name'].iloc[i].endswith(' ***'):
            classification_dataset.loc[i, 'user_name'] = classification_dataset.loc[i, 'user_name'][:-4]
            
    # Merge the two datasets into final_dataset.csv        
    merged_df = pd.merge(new_dataset, classification_dataset, on='user_name')
    merged_df=merged_df.drop(columns=['nb_emojis','id','captions'])
    merged_df.to_csv('./dataset/final_dataset.csv')




convert_mp4_to_image()
generate_dataset_from_data()
merge_datasets()