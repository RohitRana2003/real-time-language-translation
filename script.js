document.getElementById("translateButton").addEventListener("click", async function() {
    const sourceLang = document.getElementById("sourceLanguage").value;
    const targetLang = document.getElementById("targetLanguage").value;
    const inputText = document.getElementById("inputText").value;

    // Send data to the Flask backend
    const response = await fetch("file:///C:/Users/Rohit%20Singh%20Rana/TranslationProjects/src/src/translator.html", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            text: inputText,
            source: sourceLang,
            target: targetLang
        })
    });

    const data = await response.json();

    // Display the translation
    if (data.translation) {
        document.getElementById("output").innerText = data.translation;
    } else {
        document.getElementById("output").innerText = "Translation failed.";
    }
});
