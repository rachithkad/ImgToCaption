import torch 
import streamlit as st
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoTokenizer
from PIL import Image
from tqdm import tqdm
import urllib.request
from itertools import cycle
import os
import openai
import pprint

openai.api_key = "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"

feature_extractor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
tokenizer=AutoTokenizer.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}


def generate_captions(photo_description):
    model_engine = "text-davinci-002"
    prompt = (f"Imagine you have a photo that matches the following description: \"{photo_description}\".\n"
              f"Please provide three creative and fun captions to put on Instagram for this image.\n"  #Use "hashtags" instead of "captions" if you want to generate Hashtags
              f"1. \n"
              f"2. \n"
              f"3. \n")
    response = openai.Completion.create(
      engine=model_engine,
      prompt=prompt,
      max_tokens=128,
      n=1,
      stop=None,
      temperature=0.7,
    )

    captions = response.choices[0].text.strip().split("\n")
    return captions


def show_sample_images():
    sampleImages = {'First':'Football.png','Second':'Horse.png','Third':'basketball_player.jpg'} 
    
    cols = cycle(st.columns(3)) 
    for sampleImage in sampleImages.values():
        next(cols).image(sampleImage, width=200)
    for i, sampleImage in enumerate(sampleImages.values()):
        if next(cols).button("Predict Caption",key=i):
            photo_description = predict_step([sampleImage],False)
            st.subheader("Description for the Image:")
            st.write(photo_description[0])
            st.subheader("Some possible captions for the picture are:")
            captions=generate_captions(photo_description[0])
            for caption in captions:
                st.write(caption)

def image_uploader():
    with st.form("uploader"):
        images = st.file_uploader("Upload Images",accept_multiple_files=True,type=["png","jpg","jpeg"])
        submitted = st.form_submit_button("Submit")
        if submitted:
            photo_description = predict_step(images,False)
            st.subheader("Description for the Image:")
            for i,caption in enumerate(photo_description):
                st.write(caption)
            st.subheader("Some possible captions for the picture are:")
            captions=generate_captions(photo_description)
            for caption in captions:
                st.write(caption)

def images_url():
    with st.form("url"):
        urls = st.text_input('Enter URL of Images followed by comma for multiple URLs')
        images = urls.split(',')
        submitted = st.form_submit_button("Submit")
        if submitted:
            photo_description = predict_step(images,True)
            st.subheader("Description for the Image:")
            for i,caption in enumerate(photo_description):
                st.write(caption)
            st.subheader("Some possible captions for the picture are:")
            captions=generate_captions(photo_description)
            for caption in captions:
                st.write(caption)

def main():
    st.set_page_config(page_title="Image Captioning")
    st.title("Image Caption Prediction")
    st.header("Hey, I'm Rachith!")
    st.subheader('Welcome to Image Caption Prediction!')
    tab1, tab2, tab3 = st.tabs(["Sample Images", "Image from computer", "Image from URL"])
    with tab1:
        show_sample_images()
    with tab2:
        image_uploader()
    with tab3:
        images_url()

def predict_step(images_list,is_url):
    images = []
    for image in tqdm(images_list):
        if is_url:
            urllib.request.urlretrieve(image, "file.jpg")
            i_image = Image.open("file.jpg")
            st.image(i_image,width=200)

        else:
            i_image = Image.open(image)
            st.image(i_image,width=200)

        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    if is_url:
        os.remove('file.jpg')
    return preds
    
if __name__ == '__main__':
    main()