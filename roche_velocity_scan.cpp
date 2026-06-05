#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Params {
    double q;
    double Gamma;

    double mu1;
    double mu2;
    double mu1_eff;
    double yG;
};

struct State {
    double x;
    double y;
    double vx;
    double vy;
};

struct TrajectoryPoint {
    double t;
    State s;
    double distance_to_G;
};

struct IntegrationResult {
    std::vector<TrajectoryPoint> points;
    int stop_reason;
    double stop_time;
    double dt;
};

struct ConvergenceResult {
    IntegrationResult result;
    double max_error;
    int refinements;
    bool converged;
};

struct SummaryRow {
    int index;
    double v0;
    std::string trajectory_file;
    double final_dt;
    bool converged;
    double max_error;
    int refinements;
    int stop_reason;
    std::string stop_explanation;
    double stop_time;
    double final_x;
    double final_y;
    double final_vx;
    double final_vy;
    double final_distance_to_G;
    std::size_t n_points;
};

Params make_params(double q, double Gamma) {
    if (q <= 0.0) {
        throw std::runtime_error("q must be strictly positive.");
    }

    if (Gamma < 0.0 || Gamma >= 1.0) {
        throw std::runtime_error("Gamma must satisfy 0 <= Gamma < 1.");
    }

    Params p;
    p.q = q;
    p.Gamma = Gamma;

    p.mu1 = q / (1.0 + q);
    p.mu2 = 1.0 / (1.0 + q);
    p.mu1_eff = (1.0 - Gamma) * p.mu1;
    p.yG = p.mu2;

    return p;
}

double lagrange_equation_y(double y, const Params& p) {
    const double r1 = std::abs(y);
    const double r2 = std::abs(y - 1.0);

    return -p.mu1_eff * y / (r1 * r1 * r1)
           -p.mu2 * (y - 1.0) / (r2 * r2 * r2)
           +(y - p.yG);
}

double bisection_root(
    double a,
    double b,
    const Params& p,
    const std::string& name,
    double tol = 1.0e-13,
    int max_iter = 200
) {
    double fa = lagrange_equation_y(a, p);
    double fb = lagrange_equation_y(b, p);

    if (fa * fb > 0.0) {
        throw std::runtime_error("Could not bracket " + name + ".");
    }

    for (int iter = 0; iter < max_iter; ++iter) {
        const double c = 0.5 * (a + b);
        const double fc = lagrange_equation_y(c, p);

        if (std::abs(fc) < tol || 0.5 * std::abs(b - a) < tol) {
            return c;
        }

        if (fa * fc <= 0.0) {
            b = c;
            fb = fc;
        } else {
            a = c;
            fa = fc;
        }
    }

    return 0.5 * (a + b);
}

double find_left_outer_lagrange_point(const Params& p) {
    const double eps = 1.0e-10;

    double a = -2.0;
    const double b = -eps;

    while (lagrange_equation_y(a, p) > 0.0) {
        a *= 2.0;

        if (std::abs(a) > 1.0e6) {
            throw std::runtime_error("Could not bracket the donor side outer point.");
        }
    }

    return bisection_root(a, b, p, "donor side outer Lagrange point");
}

double find_inner_lagrange_point(const Params& p) {
    const double eps = 1.0e-10;
    return bisection_root(eps, 1.0 - eps, p, "inner Lagrange point L1");
}

double find_right_outer_lagrange_point(const Params& p) {
    const double eps = 1.0e-10;

    const double a = 1.0 + eps;
    double b = 2.0;

    while (lagrange_equation_y(b, p) < 0.0) {
        b *= 2.0;

        if (b > 1.0e6) {
            throw std::runtime_error("Could not bracket the accretor side outer point.");
        }
    }

    return bisection_root(a, b, p, "accretor side outer Lagrange point");
}

State rhs(const State& s, const Params& p) {
    const double dx1 = s.x;
    const double dy1 = s.y;

    const double dx2 = s.x;
    const double dy2 = s.y - 1.0;

    const double r1_sq = dx1 * dx1 + dy1 * dy1;
    const double r2_sq = dx2 * dx2 + dy2 * dy2;

    if (r1_sq == 0.0 || r2_sq == 0.0) {
        throw std::runtime_error("The particle reached one of the point masses.");
    }

    const double r1_cubed = r1_sq * std::sqrt(r1_sq);
    const double r2_cubed = r2_sq * std::sqrt(r2_sq);

    const double ax_grav =
        -p.mu1_eff * dx1 / r1_cubed
        -p.mu2 * dx2 / r2_cubed;

    const double ay_grav =
        -p.mu1_eff * dy1 / r1_cubed
        -p.mu2 * dy2 / r2_cubed;

    const double ax_cent = s.x;
    const double ay_cent = s.y - p.yG;

    const double ax_cor = 2.0 * s.vy;
    const double ay_cor = -2.0 * s.vx;

    State ds;
    ds.x = s.vx;
    ds.y = s.vy;
    ds.vx = ax_grav + ax_cent + ax_cor;
    ds.vy = ay_grav + ay_cent + ay_cor;

    return ds;
}

State add_scaled(const State& s, const State& k, double scale) {
    State out;
    out.x = s.x + scale * k.x;
    out.y = s.y + scale * k.y;
    out.vx = s.vx + scale * k.vx;
    out.vy = s.vy + scale * k.vy;
    return out;
}

State rk4_step(const State& s, double dt, const Params& p) {
    const State k1 = rhs(s, p);
    const State k2 = rhs(add_scaled(s, k1, 0.5 * dt), p);
    const State k3 = rhs(add_scaled(s, k2, 0.5 * dt), p);
    const State k4 = rhs(add_scaled(s, k3, dt), p);

    State out;
    out.x  = s.x  + dt * (k1.x  + 2.0 * k2.x  + 2.0 * k3.x  + k4.x ) / 6.0;
    out.y  = s.y  + dt * (k1.y  + 2.0 * k2.y  + 2.0 * k3.y  + k4.y ) / 6.0;
    out.vx = s.vx + dt * (k1.vx + 2.0 * k2.vx + 2.0 * k3.vx + k4.vx) / 6.0;
    out.vy = s.vy + dt * (k1.vy + 2.0 * k2.vy + 2.0 * k3.vy + k4.vy) / 6.0;

    return out;
}

double distance_to_center_of_mass(const State& s, const Params& p) {
    const double dx = s.x;
    const double dy = s.y - p.yG;
    return std::sqrt(dx * dx + dy * dy);
}

IntegrationResult integrate_trajectory(
    const Params& p,
    double L1,
    double v0,
    double dt
) {
    const double pi = std::acos(-1.0);
    const double t_max = 8.0 * pi;

    State s;
    s.x = 0.0;
    s.y = L1;
    s.vx = 0.0;
    s.vy = v0;

    double t = 0.0;

    IntegrationResult res;
    res.dt = dt;
    res.stop_reason = 2;
    res.stop_time = t_max;

    while (true) {
        const double distG = distance_to_center_of_mass(s, p);

        res.points.push_back({t, s, distG});

        if (distG >= 2.0) {
            res.stop_reason = 1;
            res.stop_time = t;
            break;
        }

        if (t >= t_max) {
            res.stop_reason = 2;
            res.stop_time = t;
            break;
        }

        double local_dt = dt;

        if (t + local_dt > t_max) {
            local_dt = t_max - t;
        }

        s = rk4_step(s, local_dt, p);
        t += local_dt;
    }

    return res;
}

State interpolate_state(
    const std::vector<TrajectoryPoint>& traj,
    double t,
    std::size_t& index
) {
    while (index + 1 < traj.size() && traj[index + 1].t < t) {
        ++index;
    }

    if (index + 1 >= traj.size()) {
        return traj.back().s;
    }

    const TrajectoryPoint& p0 = traj[index];
    const TrajectoryPoint& p1 = traj[index + 1];

    const double dt = p1.t - p0.t;

    if (dt <= 0.0) {
        return p0.s;
    }

    const double w = (t - p0.t) / dt;

    State s;
    s.x  = p0.s.x  + w * (p1.s.x  - p0.s.x);
    s.y  = p0.s.y  + w * (p1.s.y  - p0.s.y);
    s.vx = p0.s.vx + w * (p1.s.vx - p0.s.vx);
    s.vy = p0.s.vy + w * (p1.s.vy - p0.s.vy);

    return s;
}

double trajectory_error(
    const IntegrationResult& coarse,
    const IntegrationResult& fine
) {
    const double t_common = std::min(coarse.stop_time, fine.stop_time);

    std::size_t coarse_index = 0;
    double max_error = 0.0;

    for (const auto& pf : fine.points) {
        if (pf.t > t_common) {
            break;
        }

        const State sc = interpolate_state(coarse.points, pf.t, coarse_index);
        const State sf = pf.s;

        const double dx = sc.x - sf.x;
        const double dy = sc.y - sf.y;

        const double diff = std::sqrt(dx * dx + dy * dy);
        const double radius = std::sqrt(sf.x * sf.x + sf.y * sf.y);

        const double scale = std::max(1.0, radius);
        const double err = diff / scale;

        max_error = std::max(max_error, err);
    }

    return max_error;
}

bool same_stopping_behavior(
    const IntegrationResult& coarse,
    const IntegrationResult& fine
) {
    if (coarse.stop_reason != fine.stop_reason) {
        return false;
    }

    const double tolerance_time = std::max(coarse.dt, fine.dt);
    return std::abs(coarse.stop_time - fine.stop_time) <= tolerance_time;
}

ConvergenceResult integrate_until_converged(
    const Params& p,
    double L1,
    double v0,
    double initial_dt,
    double tolerance,
    int max_refinements
) {
    if (initial_dt <= 0.0) {
        throw std::runtime_error("The initial timestep must be positive.");
    }

    if (tolerance <= 0.0) {
        throw std::runtime_error("The convergence tolerance must be positive.");
    }

    IntegrationResult coarse = integrate_trajectory(p, L1, v0, initial_dt);

    double dt = 0.5 * initial_dt;
    double last_error = 1.0e99;

    for (int refinement = 1; refinement <= max_refinements; ++refinement) {
        IntegrationResult fine = integrate_trajectory(p, L1, v0, dt);

        last_error = trajectory_error(coarse, fine);
        const bool same_stop = same_stopping_behavior(coarse, fine);

        std::cout << "  convergence test " << refinement
                  << ": dt = " << dt
                  << ", max_error = " << last_error
                  << ", stop_reason = " << fine.stop_reason
                  << ", stop_time = " << fine.stop_time
                  << "\n";

        if (last_error < tolerance && same_stop) {
            return {fine, last_error, refinement, true};
        }

        coarse = fine;
        dt *= 0.5;
    }

    return {coarse, last_error, max_refinements, false};
}

std::vector<double> make_logspace_velocities(
    double v0_min,
    double v0_max,
    int n_values
) {
    if (v0_min <= 0.0 || v0_max <= 0.0) {
        throw std::runtime_error("v0_min and v0_max must be strictly positive for logarithmic spacing.");
    }

    if (v0_max < v0_min) {
        throw std::runtime_error("v0_max must be larger than or equal to v0_min.");
    }

    if (n_values <= 0) {
        throw std::runtime_error("The number of velocities must be positive.");
    }

    std::vector<double> values;
    values.reserve(n_values);

    if (n_values == 1) {
        values.push_back(v0_min);
        return values;
    }

    const double log_min = std::log(v0_min);
    const double log_max = std::log(v0_max);

    for (int i = 0; i < n_values; ++i) {
        const double f = static_cast<double>(i) / static_cast<double>(n_values - 1);
        values.push_back(std::exp(log_min + f * (log_max - log_min)));
    }

    return values;
}

std::string make_trajectory_filename(
    const std::string& prefix,
    int index
) {
    std::ostringstream name;
    name << prefix << "_trajectory_" << std::setw(2) << std::setfill('0') << index << ".csv";
    return name.str();
}

std::string stop_explanation(int stop_reason) {
    if (stop_reason == 1) {
        return "distance_to_center_of_mass_exceeded_2a";
    }

    if (stop_reason == 2) {
        return "integration_exceeded_4_orbital_periods";
    }

    return "unknown";
}

void write_trajectory_csv(
    const std::string& output_name,
    const Params& p,
    double v0,
    double L_outer_donor,
    double L1,
    double L_outer_accretor,
    double initial_dt,
    double tolerance,
    const ConvergenceResult& conv
) {
    std::ofstream out(output_name);

    if (!out) {
        throw std::runtime_error("Could not open trajectory output file.");
    }

    out << std::setprecision(16);

    out << "# Roche trajectory in dimensionless units\n";
    out << "# length_unit=a\n";
    out << "# time_unit=Omega_inverse\n";
    out << "# velocity_unit=a_Omega\n";
    out << "# q=" << p.q << "\n";
    out << "# Gamma=" << p.Gamma << "\n";
    out << "# v0=" << v0 << "\n";
    out << "# mu1=" << p.mu1 << "\n";
    out << "# mu2=" << p.mu2 << "\n";
    out << "# mu1_eff=" << p.mu1_eff << "\n";
    out << "# y_center_of_mass=" << p.yG << "\n";
    out << "# L_outer_donor=" << L_outer_donor << "\n";
    out << "# L1_inner=" << L1 << "\n";
    out << "# L_outer_accretor=" << L_outer_accretor << "\n";
    out << "# initial_dt=" << initial_dt << "\n";
    out << "# final_dt=" << conv.result.dt << "\n";
    out << "# convergence_tolerance=" << tolerance << "\n";
    out << "# convergence_max_error=" << conv.max_error << "\n";
    out << "# convergence_refinements=" << conv.refinements << "\n";
    out << "# convergence_success=" << (conv.converged ? 1 : 0) << "\n";
    out << "# stop_reason=" << conv.result.stop_reason << "\n";
    out << "# stop_explanation=" << stop_explanation(conv.result.stop_reason) << "\n";
    out << "# stop_time=" << conv.result.stop_time << "\n";

    out << "t,x,y,vx,vy,distance_to_center_of_mass\n";

    for (const auto& pnt : conv.result.points) {
        out << pnt.t << ","
            << pnt.s.x << ","
            << pnt.s.y << ","
            << pnt.s.vx << ","
            << pnt.s.vy << ","
            << pnt.distance_to_G << "\n";
    }
}

void write_summary_csv(
    const std::string& output_name,
    const Params& p,
    double v0_min,
    double v0_max,
    int n_velocities,
    double L_outer_donor,
    double L1,
    double L_outer_accretor,
    double initial_dt,
    double tolerance,
    int max_refinements,
    const std::vector<SummaryRow>& rows
) {
    std::ofstream out(output_name);

    if (!out) {
        throw std::runtime_error("Could not open summary output file.");
    }

    out << std::setprecision(16);

    out << "# Roche velocity scan summary\n";
    out << "# length_unit=a\n";
    out << "# time_unit=Omega_inverse\n";
    out << "# velocity_unit=a_Omega\n";
    out << "# q=" << p.q << "\n";
    out << "# Gamma=" << p.Gamma << "\n";
    out << "# v0_min=" << v0_min << "\n";
    out << "# v0_max=" << v0_max << "\n";
    out << "# n_velocities=" << n_velocities << "\n";
    out << "# mu1=" << p.mu1 << "\n";
    out << "# mu2=" << p.mu2 << "\n";
    out << "# mu1_eff=" << p.mu1_eff << "\n";
    out << "# y_center_of_mass=" << p.yG << "\n";
    out << "# L_outer_donor=" << L_outer_donor << "\n";
    out << "# L1_inner=" << L1 << "\n";
    out << "# L_outer_accretor=" << L_outer_accretor << "\n";
    out << "# initial_dt=" << initial_dt << "\n";
    out << "# convergence_tolerance=" << tolerance << "\n";
    out << "# max_refinements=" << max_refinements << "\n";

    out << "index,v0,trajectory_file,final_dt,converged,max_error,refinements,"
        << "stop_reason,stop_explanation,stop_time,"
        << "final_x,final_y,final_vx,final_vy,final_distance_to_center_of_mass,n_points\n";

    for (const auto& row : rows) {
        out << row.index << ","
            << row.v0 << ","
            << row.trajectory_file << ","
            << row.final_dt << ","
            << (row.converged ? 1 : 0) << ","
            << row.max_error << ","
            << row.refinements << ","
            << row.stop_reason << ","
            << row.stop_explanation << ","
            << row.stop_time << ","
            << row.final_x << ","
            << row.final_y << ","
            << row.final_vx << ","
            << row.final_vy << ","
            << row.final_distance_to_G << ","
            << row.n_points << "\n";
    }
}

int main(int argc, char* argv[]) {
    try {
        double q;
        double v0_min;
        double v0_max;
        double Gamma;

        double initial_dt = 1.0e-3;
        double tolerance = 1.0e-2;
        int max_refinements = 12;

        const int n_velocities = 5;

        std::string output_prefix = "roche";
        std::string summary_file = "roche_summary.csv";

        if (argc >= 5) {
            q = std::atof(argv[1]);
            v0_min = std::atof(argv[2]);
            v0_max = std::atof(argv[3]);
            Gamma = std::atof(argv[4]);
        } else {
            std::cout << "Mass ratio q = M1/M2: ";
            std::cin >> q;

            std::cout << "Minimum initial speed v0_min in units of a Omega: ";
            std::cin >> v0_min;

            std::cout << "Maximum initial speed v0_max in units of a Omega: ";
            std::cin >> v0_max;

            std::cout << "Eddington parameter Gamma: ";
            std::cin >> Gamma;
        }

        if (argc >= 6) {
            initial_dt = std::atof(argv[5]);
        }

        if (argc >= 7) {
            output_prefix = argv[6];
        }

        if (argc >= 8) {
            summary_file = argv[7];
        }

        if (argc >= 9) {
            max_refinements = std::atoi(argv[8]);
        }

        if (argc >= 10) {
            tolerance = std::atof(argv[9]);
        }

        const Params p = make_params(q, Gamma);

        const double L_outer_donor = find_left_outer_lagrange_point(p);
        const double L1 = find_inner_lagrange_point(p);
        const double L_outer_accretor = find_right_outer_lagrange_point(p);

        const std::vector<double> velocities = make_logspace_velocities(
            v0_min,
            v0_max,
            n_velocities
        );

        std::cout << "Computed Lagrange points along the y axis:\n";
        std::cout << "  outer point on donor side:    y = " << L_outer_donor << "\n";
        std::cout << "  inner point L1:               y = " << L1 << "\n";
        std::cout << "  outer point on accretor side: y = " << L_outer_accretor << "\n";

        std::vector<SummaryRow> summary_rows;
        summary_rows.reserve(n_velocities);

        for (int i = 0; i < n_velocities; ++i) {
            const double v0 = velocities[i];
            const std::string trajectory_file = make_trajectory_filename(output_prefix, i);

            std::cout << "\nRunning trajectory " << i
                      << " with v0 = " << v0 << "\n";

            const ConvergenceResult conv = integrate_until_converged(
                p,
                L1,
                v0,
                initial_dt,
                tolerance,
                max_refinements
            );

            write_trajectory_csv(
                trajectory_file,
                p,
                v0,
                L_outer_donor,
                L1,
                L_outer_accretor,
                initial_dt,
                tolerance,
                conv
            );

            const TrajectoryPoint& last = conv.result.points.back();

            SummaryRow row;
            row.index = i;
            row.v0 = v0;
            row.trajectory_file = trajectory_file;
            row.final_dt = conv.result.dt;
            row.converged = conv.converged;
            row.max_error = conv.max_error;
            row.refinements = conv.refinements;
            row.stop_reason = conv.result.stop_reason;
            row.stop_explanation = stop_explanation(conv.result.stop_reason);
            row.stop_time = conv.result.stop_time;
            row.final_x = last.s.x;
            row.final_y = last.s.y;
            row.final_vx = last.s.vx;
            row.final_vy = last.s.vy;
            row.final_distance_to_G = last.distance_to_G;
            row.n_points = conv.result.points.size();

            summary_rows.push_back(row);

            std::cout << "  file: " << trajectory_file << "\n";
            std::cout << "  converged: " << (conv.converged ? "yes" : "no") << "\n";
            std::cout << "  final dt: " << conv.result.dt << "\n";
            std::cout << "  max error: " << conv.max_error << "\n";
            std::cout << "  stop reason: " << conv.result.stop_reason
                      << " (" << stop_explanation(conv.result.stop_reason) << ")\n";
        }

        write_summary_csv(
            summary_file,
            p,
            v0_min,
            v0_max,
            n_velocities,
            L_outer_donor,
            L1,
            L_outer_accretor,
            initial_dt,
            tolerance,
            max_refinements,
            summary_rows
        );

        std::cout << "\nSummary written to " << summary_file << "\n";

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}