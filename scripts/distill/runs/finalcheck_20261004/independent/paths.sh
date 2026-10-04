# path_of CLIP ROW -> render path (provenance exactly as TABLE_HEADLINE_12CLIP.txt / the speed lane's tables)
path_of() {
  local c=$1 m=$2
  case $m in
    origin_ll) echo outputs/beyond4_lossless/clips/${c}_origin_ll/${c}_inpainting_results_sbs.mkv ;;
    deliv)     echo outputs/beyond_distil_mamba_scaled/clips/${c}_mstudent2_step800_deliv_ll/${c}_inpainting_results_sbs.mkv ;;
    mamba_ll)  case $c in 0052|0147|0204|0301) echo outputs/skeptic1_stack/clips/${c}_mamba_ll/${c}_inpainting_results_sbs.mkv ;;
                          *) echo outputs/beyond_distil_mamba/clips/${c}_mamba_ll/${c}_inpainting_results_sbs.mkv ;; esac ;;
    s25_ll)    case $c in 0052|0147|0204|0301) echo outputs/beyond4_lossless/clips/${c}_s25_ll/${c}_inpainting_results_sbs.mkv ;;
                          *) echo outputs/skeptic1_stack/clips/${c}_s25_ll/${c}_inpainting_results_sbs.mkv ;; esac ;;
    origin_*|deliv_*) echo outputs/finalcheck_20261004/speed/clips/${c}_${m}/${c}_inpainting_results_sbs.mkv ;;
    *) echo "BADROW_$m" ;;
  esac
}
