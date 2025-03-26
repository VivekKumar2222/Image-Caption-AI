from flask import Flask, render_template, request
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import io
import base64 

app = Flask(__name__)

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'image' not in request.files or request.files['image'].filename == '':
            return "No image uploaded", 400  

        image_file = request.files['image']  
        img = Image.open(io.BytesIO(image_file.read())).convert('RGB') 
        
        
        inputs = processor(images=img, return_tensors="pt")
        output = model.generate(**inputs)

        
        caption = processor.batch_decode(output, skip_special_tokens=True)[0]

        #
        img_io = io.BytesIO()
        img.save(img_io, format='JPEG')
        img_io.seek(0)
        img_base64 = base64.b64encode(img_io.read()).decode('utf-8')  

        return render_template('new_home.html', caption=caption, img_data=img_base64)

    return render_template('new_home.html')  # Render the form

if __name__ == '__main__':
    app.run(debug=True)
