let timer;
function makeToast(title,message) {
    closeToast();
    setTimeout(() => {
      document.getElementById("toast_title").innerHTML = title;
      document.getElementById("toast_body").innerHTML = message;
      var toast = document.getElementById("toast");
      var progress = document.querySelector(".progress");

      toast.classList.add("activeNoti");
      progress.classList.add("activeNoti");
      timer = setTimeout(() => {
        closeToast();
      }, 5000);
    }, 100);
}

function closeToast() {
  clearTimeout(timer);
  var toast = document.getElementById("toast");
  var progress = document.querySelector(".progress");
  toast.classList.remove("activeNoti");
  progress.classList.remove("activeNoti");
}

export {
    makeToast,
}