export default {
  worker: {
    name: "terasweep",
    compatibilityDate: "2026-11-24",
    domains: ["terasweep.autonom.io"],
    workersDev: false,
    previewUrls: false,
    observability: {
      enabled: true,
      traces: { enabled: true },
    },
  },
};
