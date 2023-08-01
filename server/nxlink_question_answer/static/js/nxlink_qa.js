

//click event
var when_click_search = function () {

  var element_search_button = $("#search");

  if (element_search_button.text() === "running") {
    alert("please wait this running finish.")
    return null;
  }

  //var
  var query = $("#query").val();

  var url = "NXLinkQA/query";

  element_search_button.text("running");
  $.ajax({
    async: true,
    type: "POST",
    url: url,
    data: {
      query: query,
    },
    success: function (js, status) {
      //log
      console.log(`url: ${url}, status: ${status}, js: ${js}`);
      element_search_button.text("search");

      //recall
      var element_nxlink_faq_workspace_table = $("#nxlink_faq_workspace_table");
      element_nxlink_faq_workspace_table.empty();

      element_nxlink_faq_workspace_table.append(`
      <thead>
        <tr>
          <td>score</td>
          <td>question</td>
          <td>answer</td>
          <td>filename</td>
          <td>header</td>
          <td>product</td>
        </tr>
      </thead>
      `)

      for (var i=0; i<js.result.faq_recall.length; i++)
      {
        if (i % 2 === 0) {
          element_nxlink_faq_workspace_table.append(`
              <tr class="alt">
                <td>${js.result.faq_recall[i]['score']}</td>
                <td>${js.result.faq_recall[i]['question']}</td>
                <td>${js.result.faq_recall[i]['answer']}</td>
                <td>${js.result.faq_recall[i]['filename']}</td>
                <td>${js.result.faq_recall[i]['header']}</td>
                <td>${js.result.faq_recall[i]['product']}</td>
              </tr>
            `)
        } else {
          element_nxlink_faq_workspace_table.append(`
              <tr>
                <td>${js.result.faq_recall[i]['score']}</td>
                <td>${js.result.faq_recall[i]['question']}</td>
                <td>${js.result.faq_recall[i]['answer']}</td>
                <td>${js.result.faq_recall[i]['filename']}</td>
                <td>${js.result.faq_recall[i]['header']}</td>
                <td>${js.result.faq_recall[i]['product']}</td>
              </tr>
            `)
        }
      }

      // answer
      var element_answer = $("#answer");
      element_answer.text(js.result.answer)

    },
    error: function (js, status) {
      console.log(`url: ${url}, status: ${status}, js: ${js}`);
      element_search_button.text("search");
      alert(js.message)
    }
  });
}


$(document).ready(function(){

  $("#search").click(function(){
    when_click_search();
  });

})
