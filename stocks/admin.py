from django.contrib import admin

from .models import *

#ModelAdmin Sites here
class StockAdmin(admin.ModelAdmin):
    list_display = ['name', 'abbreviation', 'quantity']

class StockPriceAdmin(admin.ModelAdmin):
    list_display = ['stock', 'opening_price', 'current_price', 'date']

class FollowedStocksAdmin(admin.ModelAdmin):
    list_display = ['user', 'stock', 'stocks_owned']
    


# Register your models here.
admin.site.register(Stock, StockAdmin)
admin.site.register(StockPrice, StockPriceAdmin)
admin.site.register(FollowedStocks, FollowedStocksAdmin)