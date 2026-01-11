const TYPE_TO_IMAGE = {
  "Red Dwarf": "assets/images/red_dwarf.jpeg",
  "Brown Dwarf": "assets/images/browndwarf.jpeg",
  "White Dwarf": "assets/images/white_dwarf.jpeg",
  "Main Sequence": "assets/images/main_sequence.jpeg",
  "Super Giants": "assets/images/supergiant.jpeg",
  "Hyper Giants": "assets/images/hypergiant.jpeg",
};

function byId(id) {
  return document.getElementById(id);
}

function setText(el, text) {
  el.textContent = text;
}

async function loadSchema() {
  const res = await fetch("/schema", { method: "GET" });
  if (!res.ok) return;

  const schema = await res.json();

  const colorSelect = byId("Color");
  const spectralSelect = byId("Spectral_Class");

  if (Array.isArray(schema.colors)) {
    colorSelect.innerHTML = "";
    for (const c of schema.colors) {
      const opt = document.createElement("option");
      opt.value = c;
      opt.textContent = c;
      colorSelect.appendChild(opt);
    }
  }

  if (Array.isArray(schema.spectral_classes)) {
    spectralSelect.innerHTML = "";
    for (const s of schema.spectral_classes) {
      const opt = document.createElement("option");
      opt.value = s;
      opt.textContent = s;
      spectralSelect.appendChild(opt);
    }
  }
}

async function predict() {
  const messageEl = byId("message");
  const resultEl = byId("result");
  const imgWrapEl = byId("imageWrap");
  const imgEl = byId("resultImage");

  setText(messageEl, "");
  setText(resultEl, "");
  imgWrapEl.classList.add("hidden");
  imgEl.removeAttribute("src");

  const tempRaw = byId("Temperature").value;
  const lRaw = byId("L").value;
  const rRaw = byId("R").value;
  const amRaw = byId("A_M").value;

  if (
    tempRaw.trim() === "" ||
    lRaw.trim() === "" ||
    rRaw.trim() === "" ||
    amRaw.trim() === ""
  ) {
    setText(messageEl, "Enter all numeric parameters before predicting.");
    return;
  }

  const payload = {
    Temperature: Number(tempRaw),
    L: Number(lRaw),
    R: Number(rRaw),
    A_M: Number(amRaw),
    Color: byId("Color").value,
    Spectral_Class: byId("Spectral_Class").value,
  };

  const res = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data = await res.json().catch(() => null);
  if (!data) {
    setText(messageEl, "Unexpected response from server.");
    return;
  }

  if (!res.ok) {
    setText(messageEl, data.message || "Validation error.");
    return;
  }

  if (data.in_domain !== true) {
    setText(messageEl, data.message || "Input is out of domain.");
    return;
  }

  const pred = data.prediction;
  const typeName = pred.type_name;
  const confidence = pred.confidence;

  setText(
    resultEl,
    `Prediction: ${typeName} (confidence ${(confidence * 100).toFixed(2)}%)`
  );

  const imgPath = TYPE_TO_IMAGE[typeName];
  if (imgPath) {
    imgEl.src = imgPath;
    imgEl.alt = typeName;
    imgWrapEl.classList.remove("hidden");
  }
}

function syncInputs(rangeId, numId) {
  const range = byId(rangeId);
  const num = byId(numId);
  if (!range || !num) return;

  range.addEventListener("input", () => {
    num.value = range.value;
  });

  num.addEventListener("input", () => {
    range.value = num.value;
  });
}

document.addEventListener("DOMContentLoaded", async () => {
  await loadSchema();
  byId("predictBtn").addEventListener("click", predict);

  syncInputs("TemperatureRange", "Temperature");
  syncInputs("LRange", "L");
  syncInputs("RRange", "R");
  syncInputs("A_MRange", "A_M");
});
