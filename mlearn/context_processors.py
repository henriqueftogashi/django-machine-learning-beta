import os

def list_processor(request):
    listfiles = os.listdir('media/downloaded')
    return {'listfiles': listfiles}
