module LEDauto (
    input CLK100MHZ,
    input btnL,
    input btnR,
    output reg[15:0] LED
);
reg [3:0] indled;
reg dir;
reg [24:0] count;
reg sposta;

initial begin
    indled = 4'b0000;
    LED = 16'b0000_0000_0000_0001;
    dir = 0;
    count = 0;
    sposta = 0;
end

always @(posedge CLK100MHZ) begin
    count = 0;
    sposta = 1;

    if (count[23] == 1) begin
        count = 0;
        sposta = 1;
    end else begin
        sposta = 0;
    end

    if(sposta) begin
        if (dir == 0) begin
            if (indled < 15) begin
                indled = indled + 1;
            end else begin
                dir = 1;
            end
        end else begin
            if (indled > 0) begin
                indled = indled - 1;
            end else begin
                dir = 0;
            end
        end
    LED = (1 << indled);
    end
end
endmodule