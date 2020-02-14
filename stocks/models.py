from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class Stock(models.Model):

    name = models.CharField(max_length=100, blank=True, null=True)
    abbreviation = models.CharField(max_length=10, blank=True, null=True)
    desc = models.TextField(blank=True, null=True)
    sold = models.IntegerField(blank=True, null=True)
    quantity = models.IntegerField(blank=True, null=True)

    class Meta:
        verbose_name = "Stock"
        verbose_name_plural = "Stocks"

    def __str__(self):
        return self.name

class StockPrice(models.Model):

    stock = models.ForeignKey("stocks.Stock", on_delete=models.CASCADE)
    date = models.DateField(auto_now=False, auto_now_add=False, blank=True, null=True)
    opening_price = models.FloatField(blank=True, null=True)
    current_price = models.FloatField(blank=True, null=True)
    
    class Meta:
        verbose_name = "StockPrice"
        verbose_name_plural = "StockPrices"

    def __str__(self):
        return str(self.stock)

class FollowedStocks(models.Model):

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    stock = models.ForeignKey("stocks.stock", on_delete=models.CASCADE)
    stocks_owned = models.IntegerField(blank=True, null=True)

    class Meta:
        verbose_name = "FollowedStocks"
        verbose_name_plural = "FollowedStocks"

    def __str__(self):
        return str(self.user)