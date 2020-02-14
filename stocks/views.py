from django.shortcuts import render, redirect
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import ListView, DetailView

from .models import Stock

# Create your views here.
class StockListView(ListView):
    model = Stock
    template_name = "stocks/stock_listview.html"

class StockDetailView(DetailView):
    model = Stock
    template_name = "stocks/stock_detailview.html"
