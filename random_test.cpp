#include "Bone.h"
#include "Skeleton.h"
#include "catmull_rom_interpolation.h"
#include "end_effectors_objective_and_gradient.h"
#include "linear_blend_skinning.h"
#include "projected_gradient_descent.h"
#include "read_model_and_rig_from_json.h"
#include "skeleton_visualization_mesh.h"
#include "transformed_tips.h"
#include <igl/LinSpaced.h>
#include <igl/get_seconds.h>
#include <igl/matlab_format.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/parula.h>
#include <igl/pinv.h>
#include <igl/project.h>
#include <igl/randperm.h>
#include <igl/unproject.h>
#include <vector>

#include <IKSolver.h>
#include <copy_skeleton_at.h>

int main(int argc, char *argv[]) {
  typedef Eigen::Map<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      MapRXd;
  igl::opengl::glfw::Viewer v;
  Eigen::MatrixXd V, U, W;
  Eigen::MatrixXi F;
  Skeleton skeleton;
  bool show_gui = false;
  // Index of selected end-effector
  int sel = -1;

  // list of indices into skeleton of bones whose tips are constrained during IK
  Eigen::VectorXi b;
  std::vector<std::vector<std::pair<double, Eigen::Vector3d>>> fk_anim;
  std::shared_ptr<IKSolver> ikSolver;
  enum IKSelectState { Idle, Selected, Done };
  IKSelectState ikSelectState = Idle;
  bool run_test = false;
  std::string test_output = "test-out.txt";
  Tolerance tolerance;
  bool test_incremental = true;
  size_t n_test_data = 1000;

  double avg_iterations = 0;
  double avg_time = 0;
  double avg_f = 0;
  double max_f = 0;
  // read mesh, skeleton and weights
  read_model_and_rig_from_json(argc > 1 ? argv[1] : "../data/robot-arm.json", V,
                               F, skeleton, W, fk_anim, b);
  auto create_solver = [=]() {
    for (int i = 1; i < argc; i++) {
      if (i < argc - 1 && strcmp(argv[i], "-s") == 0) {
        if (strcmp(argv[i + 1], "bfgs") == 0) {
          return create_BFGS_solver();
        } else if (strcmp(argv[i + 1], "gd") == 0) {
          return create_gradient_descent_solver();
        } else if (strcmp(argv[i + 1], "gauss") == 0) {
          return create_Gauss_Newton_solver();
        } else if (strcmp(argv[i + 1], "newton") == 0) {
          return create_Newton_solver();
        }
      }
    }
    return create_gradient_descent_solver();
  };
  auto parse_opt = [&]() {
    for (int i = 1; i < argc; i++) {
      if (strcmp(argv[i], "-i") == 0) {
        test_incremental = true;
      }
      if (strcmp(argv[i], "-u") == 0) {
        test_incremental = false;
      }
      if (i < argc - 1 && strcmp(argv[i], "-n") == 0) {
        n_test_data = std::stol(std::string(argv[i + 1]));
      }
      if (i < argc - 2 && strcmp(argv[i], "--tol") == 0) {
        tolerance.grad_tolerance = std::stod(std::string(argv[i + 1]));
        tolerance.f_tolerance = std::stod(std::string(argv[i + 2]));
      }
      if (strcmp(argv[i], "--gui") == 0) {
        show_gui = true;
      }
      if (strcmp(argv[i], "--no-gui") == 0) {
        show_gui = false;
      }
      if (i < argc - 1 && strcmp(argv[i], "-o") == 0) {
        test_output = argv[i + 1];
      }
    }
  };
  parse_opt();
  std::ofstream test_out_file(test_output);
  if (!fk_anim.empty()) {
    for (auto &fk : fk_anim) {
      for (auto &frame : fk) {
        printf("%lf %lf %lf %lf\n", frame.first, frame.second[0],
               frame.second[1], frame.second[2]);
      }
    }
  }
  // If not provided use last bone;
  if (b.size() == 0) {
    b.setConstant(1, 1, skeleton.size() - 1);
  }

  // endpoint positions in a single column
  Eigen::VectorXd xb0 = transformed_tips(skeleton, b);

  // U will track the deforming mesh
  U = V;
  const int model_id = 0;
  // skeleton after so dots can be on top
  const int skeleton_id = 1;
  v.append_mesh();
  v.selected_data_index = 0;
  v.data_list[model_id].set_mesh(U, F);
  v.data_list[model_id].show_faces = false;
  v.data_list[model_id].set_face_based(true);
  // Color the model based on linear blend skinning weights
  {
    Eigen::MatrixXd CM =
        (Eigen::MatrixXd(8, 3) << 228, 26, 28, 55, 126, 184, 77, 175, 74, 152,
         78, 163, 255, 127, 0, 255, 255, 51, 166, 86, 40, 247, 129, 191)
            .finished() /
        255.0;
    Eigen::MatrixXd VC =
        W * CM.replicate((W.cols() + CM.rows() - 1) / CM.rows(), 1)
                .topRows(W.cols());
    Eigen::MatrixXd FC = Eigen::MatrixXd::Zero(F.rows(), VC.cols());
    for (int i = 0; i < F.rows(); ++i)
      for (int j = 0; j < F.cols(); ++j)
        FC.row(i) += VC.row(F(i, j));
    FC.array() /= F.cols();
    v.data_list[model_id].set_colors(FC);
  }
  // Create a mesh to visualize the skeleton
  Eigen::MatrixXd SV, SC;
  Eigen::MatrixXi SF;
  const double thickness =
      0.01 * (V.colwise().maxCoeff() - V.colwise().minCoeff()).norm();
  skeleton_visualization_mesh(skeleton, thickness, SV, SF, SC);
  v.data_list[skeleton_id].set_mesh(SV, SF);
  v.data_list[skeleton_id].set_colors(SC);
  v.data_list[skeleton_id].set_face_based(true);
  v.core.animation_max_fps = 30.;
  v.core.is_animating = true;

  double anim_last_t = igl::get_seconds();
  double anim_t = 0;
  // Update the skeleton mesh and the linear blend skinning model based on
  // current skeleton deformation
  const auto update = [&]() {
    skeleton_visualization_mesh(skeleton, thickness, SV, SF, SC);
    v.data_list[skeleton_id].set_mesh(SV, SF);
    v.data_list[skeleton_id].compute_normals();
    v.data_list[skeleton_id].set_colors(SC);
    // Draw teal dots with lines attached to end-effectors and constraints (pink
    // if selected)
    const Eigen::RowVector3d teal(0.56471, 0.84706, 0.76863);
    const Eigen::RowVector3d pink(0.99608, 0.76078, 0.76078);
    {
      Eigen::MatrixXd C = teal.replicate(xb0.size() / 3, 1);
      if (sel != -1) {
        C.row(sel) = pink;
      }
      v.data_list[skeleton_id].set_points(MapRXd(xb0.data(), xb0.size() / 3, 3),
                                          C);
      Eigen::MatrixXd P(xb0.size() / 3 * 2, 3);
      Eigen::VectorXd xb = transformed_tips(skeleton, b);
      P << MapRXd(xb0.data(), xb0.size() / 3, 3),
          MapRXd(xb.data(), xb.size() / 3, 3);
      Eigen::MatrixXi E(xb.size() / 3, 2);
      for (int e = 0; e < E.rows(); e++) {
        E(e, 0) = e;
        E(e, 1) = e + E.rows();
      }
      v.data_list[skeleton_id].set_edges(P, E, C);
      v.data_list[skeleton_id].show_overlay_depth = false;
    }
    // Compute transformations of skeleton bones
    std::vector<Eigen::Affine3d, Eigen::aligned_allocator<Eigen::Affine3d>> T;
    forward_kinematics(skeleton, T);
    // Apply bone transformations to deform shape
    linear_blend_skinning(V, skeleton, T, W, U);
    v.data_list[model_id].set_vertices(U);
    v.data_list[model_id].compute_normals();
  };
  std::vector<Eigen::VectorXd, Eigen::aligned_allocator<Eigen::VectorXd>>
      test_A;

  bool test_done = false;
  const auto generate_test_data = [&](bool incremental) {
    test_A.resize(n_test_data);
    for (auto i = 0; i < n_test_data; i++) {
      if (!incremental) {
        test_A[i] = 360 * Eigen::VectorXd::Random(skeleton.size() * 3);
      } else if (incremental && i == 0) {
        test_A[i] = Eigen::VectorXd::Zero(skeleton.size() * 3);
      } else {
        test_A[i] =
            test_A[i - 1] + 5 * Eigen::VectorXd::Random(skeleton.size() * 3);
      }

      for (int b = 0; b < skeleton.size(); b++) {
        for (int k = 0; k < 3; k++) {
          auto x = test_A[i](3 * b + k);
          x = std::fmin(std::fmax(x, skeleton[b].xzx_min(k)),
                        skeleton[b].xzx_max(k));
          test_A[i](3 * b + k) = x;
        }
      }
    }
  };
  generate_test_data(test_incremental);
  int mouse_x, mouse_y;
  double mouse_z;
  bool use_ik = true;
  int test_idx = 0;
  std::vector<double> fs;
  const auto ik = [&]() {
    // If in debug mode use 1 ik iteration per drawn frame, otherwise 100
    //        const int max_iters =
    //#if NDEBUG
    //                100;
    //#else
    //        1;
    //#endif
    // Gather initial angles
    Timer frameTimer;
    while (frameTimer.elapsed_seconds() < 1.0 / 30 && run_test) {
      if (run_test) {

        {
          Eigen::VectorXd A;
          if (test_incremental && test_idx > 0) {
            A = test_A[test_idx - 1]; // set start pos to last target pos
          } else {
            A = Eigen::VectorXd::Zero(skeleton.size() * 3);
          }
          if (!ikSolver) {
            ikSolver = create_solver();
          }
          auto cp = copy_skeleton_at(skeleton, test_A[test_idx]);
          xb0 = transformed_tips(cp, b);
          ikSolver->initialize(skeleton, b, xb0, A, tolerance);
        }
        ikSelectState = Idle;
        Timer timer;
        int iter = 0;
        for (iter = 0;
             !ikSolver->has_reached_target() && timer.elapsed_seconds() < 1;
             iter++) {
          ikSolver->do_iteration();
        }
        auto elapsed = timer.elapsed_seconds();
        Eigen::VectorXd &A = ikSolver->get_z();
        for (int si = 0; si < skeleton.size(); si++) {
          skeleton[si].xzx = A.block(si * 3, 0, 3, 1);
        }

        test_idx++;
        //      printf("%d %lf %d\n", test_idx, elapsed, 0);
        test_out_file << test_idx << " " << elapsed << " " << iter << " "
                      << ikSolver->get_f() << std::endl;
        avg_iterations += iter;
        avg_time += elapsed;
        avg_f += ikSolver->get_f();
        fs.emplace_back(ikSolver->get_f());
        max_f = std::fmax(max_f, ikSolver->get_f());
        if (test_idx >= n_test_data) {
          std::sort(fs.begin(), fs.end());
          int quartile = n_test_data / 4;
          int n_remain = n_test_data - 2 * quartile;
          double interquartile =
              std::accumulate(fs.begin() + quartile, fs.end() - quartile, 0.0) / n_remain;
          avg_iterations /= n_test_data;
          avg_time /= n_test_data;
          avg_f /= n_test_data;
          run_test = false;
          test_idx = 0;
          test_done = true;
          test_out_file << avg_time << " " << avg_iterations << " " << avg_f << " "
                        << interquartile
                        << " " << max_f << std::endl;
        }
      }
    }
  };

  v.callback_pre_draw = [&](igl::opengl::glfw::Viewer &) -> bool {
    if (use_ik) {
      ik();
    } else
    // Forward Kinematice animation
    {
      if (v.core.is_animating) {
        const double now = igl::get_seconds();
        anim_t += now - anim_last_t;
        anim_last_t = now;
      }
      // Robot-arm
      for (int b = 0; b < skeleton.size(); b++) {
        skeleton[b].xzx = catmull_rom_interpolation(fk_anim[b], anim_t);
      }
    }
    update();
    return false;
  };

  // Record mouse information on click
  v.callback_mouse_down = [&](igl::opengl::glfw::Viewer &, int, int) -> bool {
    Eigen::RowVector3f last_mouse(mouse_x, v.core.viewport(3) - mouse_y, 0);
    // Move closest control point
    Eigen::MatrixXf CP;
    igl::project(MapRXd(xb0.data(), xb0.size() / 3, 3), v.core.view,
                 v.core.proj, v.core.viewport, CP);
    Eigen::VectorXf D = (CP.rowwise() - last_mouse).rowwise().norm();
    sel = (D.minCoeff(&sel) < 30) ? sel : -1;
    if (sel != -1) {
      printf("reset\n");
      ikSelectState = Selected;
      mouse_z = CP(sel, 2);
      return true;
    }
    return false;
  };
  // Unset selection on mouse up
  v.callback_mouse_up = [&](igl::opengl::glfw::Viewer &, int, int) -> bool {
    if (ikSelectState == Selected)
      ikSelectState = Done;
    sel = -1;
    return false;
  };
  // update selected constraint on mouse drag
  v.callback_mouse_move = [&](igl::opengl::glfw::Viewer &v, int _mouse_x,
                              int _mouse_y) {
    // Remember mouse position
    if (sel != -1) {
      Eigen::Vector3f drag_scene, last_scene;
      igl::unproject(
          Eigen::Vector3f(_mouse_x, v.core.viewport(3) - _mouse_y, mouse_z),
          v.core.view, v.core.proj, v.core.viewport, drag_scene);
      igl::unproject(
          Eigen::Vector3f(mouse_x, v.core.viewport(3) - mouse_y, mouse_z),
          v.core.view, v.core.proj, v.core.viewport, last_scene);
      xb0.block(sel * 3, 0, 3, 1) += (drag_scene - last_scene).cast<double>();
    }
    mouse_x = _mouse_x;
    mouse_y = _mouse_y;
    return sel != -1;
  };

  v.callback_key_pressed = [&](igl::opengl::glfw::Viewer &v, unsigned char key,
                               int /*modifier*/
                               ) -> bool {
    switch (key) {
    default:
      return false;
    case 'R':
    case 'r':
      // Reset bone transformations
      for (auto &bone : skeleton) {
        bone.xzx.setConstant(0);
      }
      sel = -1;
      xb0 = transformed_tips(skeleton, b);
      anim_last_t = igl::get_seconds();
      anim_t = 0;
      update();
      break;
    case 'I':
    case 'i':
      use_ik = !use_ik;
      v.data_list[skeleton_id].show_overlay = use_ik;
      break;
    case 'p':
    case 'P': {
      run_test = true;
    }
    case ' ':
      v.core.is_animating = !v.core.is_animating;
      if (v.core.is_animating) {
        // Reset clock
        anim_last_t = igl::get_seconds();
      }
      break;
    }
    return true;
  };
  std::cout << R"(
[space]  toggle animation
I,i      toggle between interactive demo (IK) / animation (pure FK)
R,r      reset bone transformations to rest
)";
  printf("test-incremental: %d, # test: %d, test-output:%s\n", test_incremental,
         n_test_data, test_output.c_str());
  if (show_gui)
    v.launch();
  else {
    run_test = true;
    while (!test_done) {
      ik();
    }
  }
}
