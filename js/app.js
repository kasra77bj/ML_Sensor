(function () {
  const $ = (id) => document.getElementById(id);
  const bundle = window.STRENML_MODELS;
  const MODEL_NAMES = { linear: "Linear Regression", svr: "SVR", mlp: "ANN (MLP)" };
  // Positive decimal number: "12", "12.5", ".5", "12." — no signs, letters or exponents.
  const NUMBER_RE = /^(\d+(\.\d*)?|\.\d+)$/;

  if (bundle && bundle.demo) $("demo-banner").hidden = false;

  function setError(key, msg) {
    document.querySelector(`.error[data-for="${key}"]`).textContent = msg || "";
    const el = $(key);
    if (el) el.classList.toggle("invalid", !!msg);
  }

  function readPositive(id) {
    const raw = $(id).value.trim();
    if (raw === "") return { error: "Required." };
    if (!NUMBER_RE.test(raw)) return { error: "Enter a positive number (e.g. 12.5)." };
    const v = parseFloat(raw);
    if (!(v > 0)) return { error: "Must be greater than zero." };
    return { value: v };
  }

  function validate() {
    let ok = true;
    const out = {};

    for (const id of ["model", "age"]) {
      const v = $(id).value;
      setError(id, v ? "" : "Please select a value.");
      if (!v) ok = false;
      out[id] = v;
    }

    const curing = document.querySelector('input[name="curing"]:checked');
    setError("curing", curing ? "" : "Please choose Yes or No.");
    if (!curing) ok = false;
    out.curing = curing ? Number(curing.value) : null;

    for (const id of ["resistivity", "permittivity", "temperature"]) {
      const r = readPositive(id);
      setError(id, r.error);
      if (r.error) ok = false;
      out[id] = r.value;
    }
    return ok ? out : null;
  }

  // Clear a field's error as soon as the user edits it.
  ["model", "age", "resistivity", "permittivity", "temperature"].forEach((id) =>
    $(id).addEventListener("input", () => setError(id, ""))
  );
  document.querySelectorAll('input[name="curing"]').forEach((r) =>
    r.addEventListener("change", () => setError("curing", ""))
  );

  $("form").addEventListener("submit", (e) => {
    e.preventDefault();
    const v = validate();
    if (!v) {
      $("result").textContent = "______";
      $("result-meta").textContent = "";
      return;
    }
    if (!bundle) {
      $("result-meta").textContent = "Models not found: run training/train_and_export.py.";
      return;
    }

    const features = [
      Number(v.age),
      v.curing,
      StrenML.toOhmM(v.resistivity, $("resistivity-unit").value),
      v.permittivity,
      StrenML.toCelsius(v.temperature, $("temperature-unit").value),
    ];
    const y = StrenML.predict(bundle, v.model, features);
    $("result").textContent = Math.max(0, y).toFixed(2);

    const m = bundle.models[v.model].metrics;
    $("result-meta").textContent =
      `${MODEL_NAMES[v.model]}` + (m ? ` · test R² ${m.r2.toFixed(3)}, MAE ${m.mae.toFixed(2)} MPa` : "");
  });
})();
