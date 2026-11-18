// Inference for models exported by training/train_and_export.py.
// Works in the browser (window.StrenML) and in Node (module.exports) for testing.
(function (root) {
  const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);

  const ACT = {
    relu: (v) => Math.max(0, v),
    tanh: Math.tanh,
    logistic: (v) => 1 / (1 + Math.exp(-v)),
    identity: (v) => v,
  };

  function runModel(m, x) {
    switch (m.type) {
      case "linear":
        return m.intercept + dot(m.coef, x);
      case "svr_rbf": {
        let s = m.intercept;
        for (let i = 0; i < m.support_vectors.length; i++) {
          const sv = m.support_vectors[i];
          let d2 = 0;
          for (let j = 0; j < x.length; j++) d2 += (sv[j] - x[j]) ** 2;
          s += m.dual_coef[i] * Math.exp(-m.gamma * d2);
        }
        return s;
      }
      case "mlp": {
        let a = x;
        const f = ACT[m.activation];
        m.weights.forEach((W, L) => {
          const last = L === m.weights.length - 1;
          a = m.biases[L].map((b, k) => {
            let z = b;
            for (let i = 0; i < a.length; i++) z += a[i] * W[i][k];
            return last ? z : f(z); // sklearn MLPRegressor output layer is identity
          });
        });
        return a[0];
      }
      default:
        throw new Error("Unknown model type: " + m.type);
    }
  }

  // raw: [age_days, curing(0/1), resistivity_ohm_m, permittivity, temperature_c]
  function predict(bundle, modelKey, raw) {
    const { mean, scale } = bundle.x_scaler;
    const x = raw.map((v, i) => (v - mean[i]) / scale[i]);
    const y = runModel(bundle.models[modelKey], x);
    return y * bundle.y_scaler.scale + bundle.y_scaler.mean;
  }

  // Unit conversion to the canonical units the models were trained on.
  const RESISTIVITY_TO_OHM_M = { "ohm.m": 1, "kohm.m": 1000, "ohm.cm": 0.01 };
  const toOhmM = (v, unit) => v * RESISTIVITY_TO_OHM_M[unit];
  const toCelsius = (v, unit) => (unit === "F" ? ((v - 32) * 5) / 9 : v);

  const api = { predict, toOhmM, toCelsius };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.StrenML = api;
})(typeof window !== "undefined" ? window : globalThis);
