$(window).scroll(function() {
    var scroll = $(window).scrollTop();
      $(".diagonal-bg svg line").attr("stroke-width",  ((30 + scroll/10)  + "%"));
      //30 is the starting width
      //alter the amount of growth by changing scroll/x
  });
  
  