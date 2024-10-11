import os
from flask import Flask, request, render_template_string
import torch
import scipy.io.wavfile
import numpy as np
from transformers import VitsModel, AutoTokenizer
import base64

app = Flask(__name__)

# Load the English MMS model and tokenizer
model = VitsModel.from_pretrained("facebook/mms-tts-eng")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

@app.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    text = request.form.get('text')
    if not text or len(text.strip()) == 0:
        return "No text provided or text is empty", 400

    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt")
    
    # Ensure input_ids are of type Long
    input_ids = inputs['input_ids'].long()
    attention_mask = inputs['attention_mask'].long()

    if input_ids.numel() == 0:
        return "Tokenization resulted in empty input", 400

    try:
        # Generate speech
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
    except RuntimeError as e:
        print(f"RuntimeError: {e}")
        print(f"Input shape: {input_ids.shape}")
        print(f"Attention mask shape: {attention_mask.shape}")
        return f"Error during speech generation: {str(e)}", 500

    # Get the audio data
    audio = output.waveform[0].numpy()

    # Normalize audio to 16-bit range
    audio = (audio * 32767).astype(np.int16)

    # Save the audio file using scipy
    output_path = "output.wav"
    scipy.io.wavfile.write(output_path, rate=model.config.sampling_rate, data=audio)

    # Read the file and encode to base64
    with open(output_path, "rb") as audio_file:
        audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')

    return audio_base64

@app.route('/')
def home():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
        <head>
            <title>Text-to-Speech API</title>
            <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
            <script>
                $(document).ready(function() {
                    $('form').on('submit', function(e) {
                        e.preventDefault();
                        $.ajax({
                            url: '/text-to-speech',
                            method: 'POST',
                            data: $('form').serialize(),
                            success: function(response) {
                                $('#audio').attr('src', 'data:audio/wav;base64,' + response);
                                $('#audio-container').show();
                            },
                            error: function(xhr, status, error) {
                                alert('Error: ' + xhr.responseText);
                            }
                        });
                    });
                });
            </script>
        </head>
        <body>
            <h1>Text-to-Speech API</h1>
            <form action="/text-to-speech" method="post">
                <textarea name="text" rows="4" cols="50"></textarea><br>
                <input type="submit" value="Convert to Speech">
            </form>
            <div id="audio-container" style="display:none;">
                <audio id="audio" controls></audio>
            </div>
        </body>
    </html>
    ''')

if __name__ == '__main__':
    app.run(debug=True)