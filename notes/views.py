from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.http import HttpResponse, Http404
from django.conf.urls import url
from django.contrib.auth.models import User

# views
from django.views.generic import (
    ListView, 
    DetailView, 
    CreateView, 
    UpdateView, 
    DeleteView
)

# models
from .models import Notes_Model
from django.contrib.auth.decorators import login_required

# forms
from .forms import UploadFileForm
from django.contrib import messages

# data filter
import operator
from django.db.models import Q
from django import forms

# file transfer
import os
from django.conf import settings
from django.core.files import File

# filters 
from .filters import NotesFilter

def home(request):
    """
    Displays home page requests.
    """
    context = {'posts': Notes_Model.objects.all}
    return render(request, template_name='home.html', context=context)


def about(request):
    """
    Displays about page requests.
    """
    title = {'title': 'About'} 
    return render(request, template_name='about.html', context=title)
    
class PostListView(ListView):
    """
    Orders the homepage in a better (list) view
    """
    model = Notes_Model
    template_name = 'home.html'
    context_object_name = 'posts'
    ordering = ['-date_posted']
    
    # Paginates the homepage by two posts per page
    paginate_by = 5

class PostDetailView(DetailView):
    model = Notes_Model

class PostCreateView(LoginRequiredMixin, CreateView):
    """
    Lets user upload files under the UPLOAD navigation button
    """
    model = Notes_Model
    fields = ['title', 'description', 'file_semester', 'branch_choice', 'file']

    @login_required
    def upload_file(self, request):
        if request.method == 'POST':
            form = UploadFileForm(request.POST, request.FILES)
            if form.is_valid():
                form.save()
                messages.success(request, f'Your files have been uploaded')
                return redirect('notes-home')

    def form_valid(self, form):
        form.instance.uploader = self.request.user
        return super().form_valid(form)
class PostUpdateView(LoginRequiredMixin, UserPassesTestMixin, UpdateView):
    """
    Lets user update a post
    """
    model = Notes_Model
    fields = ['title', 'description', 'file_semester', 'branch_choice', 'file']

    @login_required
    def upload_file(self, request):
        if request.method == 'POST':
            form = UploadFileForm(request.POST, request.FILES)
            if form.is_valid():
                form.save()
                return redirect('notes-detail')
    
    def test_func(self):
        post = self.get_object()
        if self.request.user == post.uploader:
            return True
        return False

class PostDeleteView(UserPassesTestMixin, LoginRequiredMixin, DeleteView):
    """
    Deletes authenticated user posts.
    """
    model = Notes_Model
    success_url = '/'

    def test_func(self):
        """
        change second self.request.user to document.uploader
        """
        post = self.get_object()
        if (self.request.user == self.request.user):
            return True
        return False


branch_choice = [
    ('Computer Science Engineering' ),
    ('Civil Engineering'),
    ('Automobile Engineering'),
    ('Electronics and Communication Engineering'),
    ('Electrical Engineering'),
    ('Electronics and Instrumentation'),
    ('Electrical and Electronics Engineering'),
    ('Fire and Safety'),
    ('Information Technology'),
    ('Mechanical Engineering'),
]

def search(request):
    branch_list = NotesFilter.objects.all()
    branch_choice_filter = NotesFilter(request.GET, queryset=branch_list)
    return render(request, 'notes/notes_model_filter.html', {'filter': branch_choice_filter})

def download(request, path):
    file_path = os.path.join(settings.MEDIA_ROOT, path)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type=None)
            response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file_path)
            return response
        raise Http404