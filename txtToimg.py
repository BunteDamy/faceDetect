from PIL import Image
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch
print(torch.cuda.is_available())
model_id1 = "stabilityai/stable-diffusion-2-1-base"
scheduler1 = EulerAncestralDiscreteScheduler.from_pretrained(model_id1, subfolder="scheduler")
pipe1 = StableDiffusionPipeline.from_pretrained(model_id1, scheduler=scheduler1, torch_dtype=torch.float16)
pipe1 = pipe1.to("cuda") # boru hattı. verileri alacak ve bir resim üretecek.

prompt1 = input("enter an english prompt:")
image1 = pipe1(prompt1).images[0]
image1.save("ading.png")

im1 = Image.open("ading.png")
im1.show()

# -------------------------------------------- OPENJOURNEY MODEL  ----------------------------------------------------------


model_id2 = "prompthero/openjourney"
pipe2 = StableDiffusionPipeline.from_pretrained(model_id2, torch_dtype=torch.float16)
pipe2 = pipe2.to("cuda")
prompt = input("enter an english prompt:")
image2 = pipe2(prompt).images[0]
image2.save("ojimg.png")
im2 = Image.open("ojimg.png")
im2.show()


# ------------------------------------------- DİL DESTEĞİ ---------------------------------------------------------

from deep_translator import GoogleTranslator

translator = GoogleTranslator(source='auto', target='en')

promptnew = translator.translate(input("Bir açıklama giriniz:"))
print(promptnew)

#stable diffusion
image11 = pipe1(promptnew).images[0]
image11.save("image11.png")
im11 = Image.open("image11.png")
print("<----------STABLE DIFFUSION-------->")
im11.show()

#open journey
image22 = pipe2(promptnew).images[0]
image22.save("image22.png")
im22 = Image.open("image22.png")
print("<----------OPEN JOURNEY--------->")
im22.show()