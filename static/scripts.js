document.addEventListener("DOMContentLoaded", function () {
  // Initialize Swiper slider
  var swiper = new Swiper(".swiper-container", {
    loop: true,
    pagination: {
      el: ".swiper-pagination",
      clickable: true,
    },
    navigation: {
      nextEl: ".swiper-button-next",
      prevEl: ".swiper-button-prev",
    },
    autoplay: {
      delay: 3000,
      disableOnInteraction: false,
    },
  });

  // Toggle dropdown menu on small screens
  const menuBtn = document.getElementById("menu-btn");
  const navbar = document.querySelector(".navbar");
  menuBtn.addEventListener("click", function () {
    navbar.classList.toggle("active");
  });

  // Active page detection (robust version)
  const links = document.querySelectorAll(".navbar a");
  // Extract the current page filename using regex
  let currentPageMatch = window.location.pathname.match(/([^\/]+)$/);
  let currentPage = currentPageMatch ? currentPageMatch[1] : "";
  // If no extension is present, default to "index.html"
  if (!currentPage || !currentPage.includes(".html")) {
    currentPage = "index.html";
  }
  console.log("Current page:", currentPage);

  links.forEach(link => {
    // Get only the filename from each link's href
    let href = link.getAttribute("href");
    // Optionally, force lower case for comparison
    if (href.toLowerCase() === currentPage.toLowerCase()) {
      link.classList.add("active");
    }
  });

  // // 🔹 Fix: Run classifyWaste() when the form is submitted
  // document.getElementById("predictForm").addEventListener("submit", function (event) {
  //   event.preventDefault(); // Stop page reload
  //   classifyWaste();
  // });
});

// Function to preview the uploaded image (used in classify page)
function previewImage() {
  let fileInput = document.getElementById("fileInput");
  let file = fileInput.files[0];
  let imagePreview = document.getElementById("imagePreview");

  if (file) {
    let reader = new FileReader();
    reader.onload = function (e) {
      imagePreview.innerHTML = `<img src="${e.target.result}" alt="Uploaded Image" style="max-width: 300px; margin-top: 10px;">`;
    };
    reader.readAsDataURL(file);
  } else {
    imagePreview.innerHTML = "";
  }
}

// Function to send the image for classification and display tips accordingly
function classifyWaste() {
  let fileInput = document.getElementById("fileInput");
  let file = fileInput.files[0];

  if (!file) {
    alert("Please upload an image first.");
    return;
  }

  let formData = new FormData();
  formData.append("file", file);

  fetch("http://127.0.0.1:5000/predict", {
    method: "POST",
    body: formData,
  })
    .then((response) => response.json())
    .then((data) => {
      let resultElement = document.getElementById("result");
      let tipsElement = document.getElementById("tips");

      resultElement.innerText = `Prediction: ${data.prediction}`;
      console.log("Before:", tipsElement.style.display);
      tipsElement.classList.remove("hidden");
      tipsElement.style.display = "block";  // Ensure it is visible
      console.log("After:", tipsElement.style.display);

      if (data.prediction === "Recyclable") {
        tipsElement.innerHTML = `
          <h3>Recycling Tips:</h3>
          <ul>
            <li>Separate plastics, paper, and metal before recycling.</li>
            <li>Rinse food containers before placing them in the recycling bin.</li>
            <li>Check with local recycling programs for guidelines.</li>
          </ul>
        `;
      } else if (data.prediction === "Organic") {
        tipsElement.innerHTML = `
          <h3>Organic Waste Handling Tips:</h3>
          <ul>
            <li>Compost fruit and vegetable scraps to create natural fertilizer.</li>
            <li>Avoid throwing meat or dairy products into compost bins.</li>
            <li>Use biodegradable waste bags for disposal.</li>
          </ul>
        `;
      }
    })
    .catch((error) => console.error("Error:", error));
}  