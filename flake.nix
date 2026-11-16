{
  description = "splats4D dev shell: viewer (Node/Vite), converter (Rust), headless browser for viewer checks";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { nixpkgs, ... }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAll = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forAll (pkgs: {
        default = pkgs.mkShell {
          packages = with pkgs; [ nodejs_22 cargo rustc ]
            ++ pkgs.lib.optionals pkgs.stdenv.hostPlatform.isLinux [ chromium ];
        };
      });
    };
}
