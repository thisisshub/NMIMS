from django.db import models
from django.contrib.auth.admin import User
from django.urls import reverse

branch_choices = [
    ('Computer Science Engineering', 'Computer Science Engineering' ),
    ('Civil Engineering', 'Civil Engineering'),
    ('Automobile Engineering', 'Automobile Engineering'),
    ('Electronics and Communication Engineering', 'Electronics and Communication Engineering'),
    ('Electrical Engineering', 'Electrical Engineering'),
    ('Electronics Engineering', 'Electronics and Instrumentation'),
    ('Electrical Engineering', 'Electrical and Electronics Engineering'),
    ('Fire Engineering', 'Fire and Safety'),
    ('Information Engineering', 'Information Technology'),
    ('Mechanical Engineering', 'Mechanical Engineering'),
]

# notes semester choice from 1 to 8
semester_choice = [tuple([x,x]) for x in range(1,9)]
class Notes_Model(models.Model):
    uploader = models.CharField(max_length=200)
    title = models.CharField(max_length=200)
    date_posted = models.DateTimeField(auto_now_add=True)
    description = models.TextField(max_length=2500)
    branch_choice = models.CharField(max_length=50, choices=branch_choices, default='cse')
    file_semester = models.IntegerField( choices=semester_choice)
    file = models.FileField()
    

    def __str__(self):
        return self.title

    def get_absolute_url(self):
        return reverse("notes-detail", kwargs={"pk": self.pk})