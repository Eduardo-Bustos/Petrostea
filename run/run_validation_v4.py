from validation.validation_engine_v4 import (
    DataSourceConfig,
    ValidationV4Config,
    run_validation_v4_bundle,
)


def main():
    ds_cfg = DataSourceConfig(
        dataset_master_path="data/processed/master/dataset_master_v50_operational.xlsx",
        merged_master_path="data/processed/master/master_v50_merged_corrected.xlsx",
        state_metrics_path="data/processed/master/Table_1_state_metrics.xlsx",
        transmission_window_path="data/processed/master/Table_2_market_transmission_window.xlsx",
        event_paths=[
            "data/processed/master/event_study_sg_v41-1.xlsx",
            "data/processed/master/event_study_rstar.xlsx",
            "data/processed/master/event_study_ssi_extended.xlsx",
            "data/processed/master/event_study_theta_collapse.xlsx",
            "data/processed/master/event_study_fai_v41.xlsx",
        ],
    )

    cfg = ValidationV4Config(
        output_dir="outputs/validation_v4"
    )

    report = run_validation_v4_bundle(ds_cfg, cfg)

    print("\n=== VALIDATION RESULT ===")
    print(report)


if __name__ == "__main__":
    main()
