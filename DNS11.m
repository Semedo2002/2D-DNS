% =========================================================================
% MASTER DNS SOLVER: 2D INCOMPRESSIBLE NAVIER-STOKES
% Method: Spectral-Galerkin with 2/3 Rule De-aliasing
% Validation Case: Taylor-Green Vortex Decay

function DNS_Master_Solver
    clearvars; clc; close all;

    % --- DOMAIN ---
    N = 128;              
    L = 2*pi;
    nu = 0.005;
    t_final = 1000.0;
    
    % ---OPERATORS---
    k_vec = [0:N/2-1 -N/2:-1]; 
    [Kx, Ky] = meshgrid(k_vec, k_vec);
    
    % (-k^2)
    K2 = Kx.^2 + Ky.^2;
    K2(1,1) = 1e-6;
    
    % de-ali
    k_max = N/2;
    mask = (abs(Kx) < 2/3 * k_max) & (abs(Ky) < 2/3 * k_max); 

    % ---ICs---
    fprintf('Initializing Taylor-Green Vortex Field...\n');
    
    dx = L/N;
    x = (0:N-1)*dx; 
    [X, Y] = meshgrid(x, x);
    
    % Deterministic IC
    w = -2 * cos(X) .* cos(Y);
    % Tinyy perturbation
    w = w + 0.02 * (rand(N) - 0.5); 
    
    w_hat = fft2(w);
    
    %Plot
    fig = figure('Units','normalized','Position',[0.2 0.2 0.6 0.6], 'Color', 'w');
    ax = axes('Parent', fig);
    Visualize(w, 0, ax);
    
    fprintf('Initialization Complete. Starting Integration...\n');

    % ---LOOP Kutta---
    t = 0;
    iter = 0;

    dt = 0.001; 
    
    while t < t_final

        if mod(iter, 10) == 0
             psi_check = -w_hat ./ K2; psi_check(1,1)=0;
             u_check = real(ifft2(1i*Ky.*psi_check));
             v_check = real(ifft2(-1i*Kx.*psi_check));
             max_vel = max(abs(u_check(:)) + abs(v_check(:))) + 1e-6;

             target_dt = 0.5 * (dx / max_vel);

             dt = 0.9 * dt + 0.1 * target_dt;
        end
        
        % ---RK4---
        dw1 = ComputeRHS(w_hat, Kx, Ky, K2, mask, nu);
        w2 = w_hat + 0.5 * dt * dw1;
        
        dw2 = ComputeRHS(w2, Kx, Ky, K2, mask, nu);
        w3 = w_hat + 0.5 * dt * dw2;
        
        dw3 = ComputeRHS(w3, Kx, Ky, K2, mask, nu);
        w4 = w_hat + dt * dw3;
        
        dw4 = ComputeRHS(w4, Kx, Ky, K2, mask, nu);

        w_hat = w_hat + (dt/6) * (dw1 + 2*dw2 + 2*dw3 + dw4);
        w_hat = w_hat .* mask; % Enforce De-aliasing
        
        t = t + dt;
        iter = iter + 1;
        
        % ---VIZ ---
        if mod(iter, 50) == 0
            w_real = real(ifft2(w_hat));

            if any(isnan(w_real(:)))
                 fprintf('Instability detected at t=%.3f. Attempting rescue...\n', t);
                 break; 
            end
            
            Visualize(w_real, t, ax);
            fprintf('Iter: %d | t: %.3f | dt: %.4f \n', iter, t, dt);
        end
    end
end

% -------------------------------------------------------------------------
% HELPER FUNCTIONS
% -------------------------------------------------------------------------

function rhs = ComputeRHS(w_hat, Kx, Ky, K2, mask, nu)
    % 1. Recover Streamfunction
    psi_hat = -w_hat ./ K2;
    psi_hat(1,1) = 0; 
    
    % 2. Derivs Fourier Space
    u_hat =  1i * Ky .* psi_hat;
    v_hat = -1i * Kx .* psi_hat;
    dw_dx_hat = 1i * Kx .* w_hat;
    dw_dy_hat = 1i * Ky .* w_hat;
    
    % 3. Real Space
    u = real(ifft2(u_hat));
    v = real(ifft2(v_hat));
    dw_dx = real(ifft2(dw_dx_hat));
    dw_dy = real(ifft2(dw_dy_hat));
    
    convection = u .* dw_dx + v .* dw_dy;
    
    % 4. de-ali
    conv_hat = fft2(convection) .* mask;
    
    % 5. RHS
    diff_hat = -nu * K2 .* w_hat;
    rhs = -conv_hat + diff_hat;
end

function Visualize(w, t, ax)
    max_val = max(abs(w(:)));
    if max_val < 1e-6, max_val = 1e-6; end 
    
    imagesc(ax, w);
    colormap(ax, jet(256));
    shading(ax, 'interp'); 
    axis(ax, 'square'); axis(ax, 'off');
    
    caxis(ax, [-max_val, max_val]); 
    colorbar(ax);
    
    title(ax, sprintf('DNS Vorticity Field | t = %.2f', t), 'FontSize', 14);
    drawnow;

end
