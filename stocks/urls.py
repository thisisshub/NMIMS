from django.urls import path

from .views import *

app_name = 'stocks'

urlpatterns = [
    path("", StockListView.as_view(), name="stock_listview"),
    path("<int:pk>/", StockDetailView.as_view(), name="stock_detailview"),
]
