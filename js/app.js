(function () {
  const $ = (id) => document.getElementById(id);
  const bundle = window.STRENML_MODELS;
  const MODELS = { linear: "Linear Regression", svr: "SVR", mlp: "ANN (MLP)" };
  const RES_UNIT_LABEL = { "ohm.m": "Ω·m", "kohm.m": "kΩ·m", "ohm.cm": "Ω·cm" };
  // Positive decimal number: "12", "12.5", ".5", "12." (no signs, letters or exponents).
  const NUMBER_RE = /^(\d+(\.\d*)?|\.\d+)$/;
  const RADIOS = ["model", "age", "curing"];
  const NUMBERS = ["resistivity", "permittivity", "temperature"];

  if (bundle && bundle.demo) $("demo-banner").hidden = false;

  const radioValue = (name) => {
    const r = document.querySelector(`input[name="${name}"]:checked`);
    return r ? r.value : null;
  };

  function setError(key, msg) {
    document.querySelector(`.error[data-for="${key}"]`).textContent = msg || "";
    const el = $(key);
    const box = el.closest(".input-group") || el;
    box.classList.toggle("invalid", !!msg);
  }

  function readPositive(id) {
    const raw = $(id).value.trim();
    if (raw === "") return { error: "Enter a value." };
    if (!NUMBER_RE.test(raw)) return { error: "Use a positive number, like 12.5." };
    const v = parseFloat(raw);
    if (!(v > 0)) return { error: "The value must be greater than zero." };
    return { value: v };
  }

  function validate() {
    let ok = true;
    const out = {};
    const msgs = { model: "Choose a model.", age: "Choose the concrete age.", curing: "Choose Yes or No." };
    for (const name of RADIOS) {
      const v = radioValue(name);
      setError(name, v ? "" : msgs[name]);
      if (!v) ok = false;
      out[name] = v;
    }
    for (const id of NUMBERS) {
      const r = readPositive(id);
      setError(id, r.error);
      if (r.error) ok = false;
      out[id] = r.value;
    }
    return ok ? out : null;
  }

  RADIOS.forEach((name) =>
    document.querySelectorAll(`input[name="${name}"]`).forEach((r) =>
      r.addEventListener("change", () => setError(name, ""))
    )
  );
  NUMBERS.forEach((id) => $(id).addEventListener("input", () => setError(id, "")));

  function resetResult(msg) {
    $("result").textContent = "—";
    $("result-meta").textContent = msg;
    $("result-r2").hidden = true;
    $("details").hidden = true;
  }

  function render(v) {
    const ohmM = StrenML.toOhmM(v.resistivity, $("resistivity-unit").value);
    const tempC = StrenML.toCelsius(v.temperature, $("temperature-unit").value);
    const features = [Number(v.age), Number(v.curing), ohmM, v.permittivity, tempC];

    const preds = {};
    for (const k of Object.keys(MODELS)) preds[k] = Math.max(0, StrenML.predict(bundle, k, features));

    const big = $("result").parentElement;
    $("result").textContent = preds[v.model].toFixed(2);
    big.classList.remove("pulse"); void big.offsetWidth; big.classList.add("pulse");

    const r2 = (k) => (bundle.models[k].metrics ? bundle.models[k].metrics.r2 * 100 : null);
    $("result-meta").textContent = `Model: ${MODELS[v.model]}`;
    const sel = r2(v.model);
    $("result-r2").innerHTML = sel == null ? "" : `Accuracy (R²): <b>${sel.toFixed(1)}%</b>`;
    $("result-r2").hidden = sel == null;

    const fmt = (n) => (Math.abs(n) >= 1000 ? n.toFixed(0) : n.toPrecision(4).replace(/\.?0+$/, ""));
    const resUnit = $("resistivity-unit").value;
    const rows = [
      ["Concrete age", `${v.age} day${v.age === "1" ? "" : "s"}`],
      ["Curing", v.curing === "1" ? "Applied" : "Not applied"],
      ["Resistivity", `${fmt(ohmM)} Ω·m` + (resUnit !== "ohm.m" ? ` (from ${fmt(v.resistivity)} ${RES_UNIT_LABEL[resUnit]})` : "")],
      ["Permittivity", fmt(v.permittivity)],
      ["Temperature", `${tempC.toFixed(1)} °C` + ($("temperature-unit").value === "F" ? ` (from ${fmt(v.temperature)} °F)` : "")],
    ];
    $("inputs-used").innerHTML = rows.map(([k, val]) => `<dt>${k}</dt><dd>${val}</dd>`).join("");

    const keys = Object.keys(MODELS);
    const best = keys.reduce((a, b) => ((r2(b) ?? -1) > (r2(a) ?? -1) ? b : a));
    $("compare").innerHTML = keys.map((k) => {
      const acc = r2(k);
      const width = acc == null ? 0 : Math.max(0, Math.min(100, acc));
      return `
      <div class="bar-row${k === v.model ? " active" : ""}">
        <div class="bar-head">
          <span class="name">${MODELS[k]}${k === best ? '<span class="chip">Most accurate</span>' : ""}</span>
          <span class="val">${preds[k].toFixed(2)} <small>MPa</small></span>
        </div>
        <div class="bar-line">
          <span class="bar-track"><span class="bar-fill" style="width:${width}%"></span></span>
          <span class="acc">${acc == null ? "n/a" : acc.toFixed(1) + "%"}</span>
        </div>
      </div>`;
    }).join("");
    $("details").hidden = false;
  }

  $("form").addEventListener("submit", (e) => {
    e.preventDefault();
    if (!bundle) return resetResult("Models not found. Run training/train_and_export.py.");
    const v = validate();
    if (!v) return resetResult("Fix the highlighted fields, then press Calculate.");
    render(v);
  });

  $("form").addEventListener("reset", () => {
    setTimeout(() => {
      [...RADIOS, ...NUMBERS].forEach((k) => setError(k, ""));
      resetResult("Fill in the form and press Calculate.");
    });
  });

  $("example").addEventListener("click", () => {
    const pick = (name, val) => (document.querySelector(`input[name="${name}"][value="${val}"]`).checked = true);
    if (!radioValue("model")) pick("model", "mlp");
    pick("age", "7");
    pick("curing", "1");
    $("resistivity").value = "45.2";
    $("resistivity-unit").value = "ohm.m";
    $("permittivity").value = "50.5";
    $("temperature").value = "23";
    $("temperature-unit").value = "C";
    $("form").requestSubmit();
  });
})();
