from django.shortcuts import render
from django.core.files.storage import FileSystemStorage

# from .predictEmotion import emotionPred
import os

cwd = os.getcwd()


def home(request):
    if request.method == 'POST' and request.FILES['myfile']:

        myfile = request.FILES['myfile']
        print(myfile.size)
        print(myfile.content_type)

        if myfile.content_type.split("/")[1] != "vnd.ms-excel":
            return render(request, 'core/emotionPrediction.html', {
                'error_file': "Error : Please Upload a CSV File",
                'uploaded_file_url': ""
            })
        if myfile.size > 23068672:
            return render(request, 'core/emotionPrediction.html', {
                'error_file': "Error : File size Exceeded 25 MB",
                'uploaded_file_url': ""
            })

        try:
            fs = FileSystemStorage()
            filename = fs.save("dataFile.csv", myfile)
            print("Filename: ", filename)
            uploaded_file_url = fs.url(filename)
            print("Uploaded file URL: ", uploaded_file_url)
            # output_pred = emotionPred(uploaded_file_url)
            output_pred = "Text"
            os.remove(cwd + uploaded_file_url)
            # attach_file_name = output_file
            # Open the file as binary mode
            try:
                print("sent")
            except:
                return render(request, 'core/emotionPrediction.html', {
                    'error_file': "Error : Email Not Sent",
                    'uploaded_file_url': output_pred
                })
            return render(request, 'core/emotionPrediction.html', {
                'error_file': output_pred,
                # 'uploaded_file_url': output_file
            })
        except Exception as e:
            return render(request, 'core/emotionPrediction.html', {
                # 'error_file': "Error : Some Error Occured",
                'error_file': str(e),
                'uploaded_file_url': ""
            })
    return render(request, 'core/emotionPrediction.html', {
        'uploaded_file_url': ""
    })
