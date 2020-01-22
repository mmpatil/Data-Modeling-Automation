const callback = () => {
  const fileInput = document.querySelector('input[type=file]');
	if (fileInput !== null) {
    fileInput.onchange = () => {
      if (fileInput.files.length > 0) {
        const fileName = document.querySelector('.file-name');
        fileName.textContent = fileInput.files[0].name;
      }
    }
  }
  const cardToggles = document.getElementsByClassName('card-toggle');
  for (let i = 0; i < cardToggles.length; i++) {
    cardToggles[i].addEventListener('click', e => {
      //TODO: this might break later (TEST)
      e.currentTarget.parentElement.parentElement.childNodes[1].classList.toggle('is-hidden');
    });
  }

  const $navbarBurgers = Array.prototype.slice.call(document.querySelectorAll('.navbar-burger'), 0);

  // Check if there are any navbar burgers
  if ($navbarBurgers.length > 0) {

    // Add a click event on each of them
    $navbarBurgers.forEach( el => {
      el.addEventListener('click', () => {

        // Get the target from the "data-target" attribute
        const target = el.dataset.target;
        const $target = document.getElementById(target);

        // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
        el.classList.toggle('is-active');
        $target.classList.toggle('is-active');

      });
    });
  }
};


if (
    document.readyState === "complete" ||
    (document.readyState !== "loading" && !document.documentElement.doScroll)
) {
  callback();
} else {
  document.addEventListener("DOMContentLoaded", callback);
}

function addExclusions() {
  var newGroup = $("#addExcl");
  $("#addExcl").append('<div class="control input-group form-group date has-icons-right datepicker"><input class="input form-control" type="text" name="exclusions" value="2006-09-30" required><span class="icon is-small is-right input-group-addon"><i class="fas fa-calendar"></i><span class="count"></span></span></div>');
  bindDatepicker();
}

function deleteExclusion() {
  $(".form-control").last().remove()
}

function addShortList() {
  var input = document.createElement("input");
  input.setAttribute("type","text")
  input.setAttribute("name", "short_list")
  input.setAttribute("class", "input short_list")
  document.getElementById("shortlist").appendChild(input)
}
function deleteShortList() {
  $(".short_list").last().remove()
}

function addBackDate() {
  var input = document.createElement("input");
  input.setAttribute("type","date")
  input.setAttribute("name", "backtest_dates")
  input.setAttribute("class", "input backtest_dates")
  document.getElementById("backdates").appendChild(input)
}

function deleteBackDate() {
  $(".backtest_dates").last().remove()
}

function deleteBackDateLong() {
  $(".backtest_long_dates").last().remove()
}

function addBackDateLong() {
  var input = document.createElement("input");
  input.setAttribute("type","date")
  input.setAttribute("name", "backtest_long_dates")
  input.setAttribute("class", "input backtest_long_dates")
  // input.setAttribute("showClearButton ", "false")
  document.getElementById("backdateLong").appendChild(input)
}

function bindDatepicker() {
  $('.datepicker').each(function(i, d) {
    $(d).datepicker({
        multidate: true,
        format: "yyyy-mm-dd",
        daysOfWeekHighlighted: "5,6",
        language: 'en'
    }).on('changeDate', function(e) {
        // `e` here contains the extra attributes
        $(this).find('.input-group-addon .count').text(', ' + e.dates.length);
    });
  })
}

function selectAll(){
  var checkboxes = document.getElementsByClassName('candidate')
  for (var i = 0; i < checkboxes.length; i++){
    if(checkboxes[i].type == 'checkbox'){
      checkboxes[i].checked = true;
    }
  }
}


$(document).ready(function() {
    bindDatepicker()
});
