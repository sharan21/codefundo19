
  $(document).ready(function() {
  $("#btn").click(function(e){
     var jsonData = {};

   var formData = $("#myform").serializeArray();
  // console.log(formData);

   $.each(formData, function() {
        if (jsonData[this.name]) {
           if (!jsonData[this.name].push) {
               jsonData[this.name] = [jsonData[this.name]];
           }
           jsonData[this.name].push(this.value || '');
       } else {
           jsonData[this.name] = this.value || '';
       }


   });
   console.log(jsonData);
    saveText(JSON.stringify(jsonData), "info.json")
//     $.ajax(
//     {
//         url : "action.php",
//         type: "POST",
//         data : jsonData,

//     });
//     e.preventDefault(); 
});
});
function saveText(text, filename){
    var a = document.createElement('a');
    a.setAttribute('href', 'data:text/plain;charset=utf-u,'+encodeURIComponent(text));
    a.setAttribute('download', filename);
    a.click()
  }
// var obj = {a: "Hello", b: "World"};
// saveText( JSON.stringify(obj), "./filename.json" );