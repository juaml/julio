$(document).ready(function() {
    hljs.highlightAll();
});


function copyToClipboard(id) {
    navigator.clipboard.writeText($(id).text())
        .then(() => {
            $("#cb-copy-success").removeClass("invisible").addClass("visible animate-pulse");
            setTimeout(function () {
                $("#cb-copy-success").addClass("invisible").removeClass("visible animate-pulse");
            }, 2000);
        })
        .catch(() => {
            $("#cb-copy-failure").removeClass("invisible").addClass("visible animate-pulse");
            setTimeout(function () {
                $("#cb-copy-failure").addClass("invisible").removeClass("visible animate-pulse");
            }, 2000);
        });
}

// From: https://stackoverflow.com/questions/19327749/javascript-blob-filename-without-link
function saveFile(name, id) {
    const data = $(id).text();
    if (data !== null && navigator.msSaveBlob)
        return navigator.msSaveBlob(new Blob([data], { type: "data:application/yaml" }), name);
    var a = $("<a style='display: none;'/>");
    var url = window.URL.createObjectURL(new Blob([data], {type: "data:application/yaml"}));
    a.attr("href", url);
    a.attr("download", name);
    $("body").append(a);
    a[0].click();
    window.URL.revokeObjectURL(url);
    a.remove();
}
