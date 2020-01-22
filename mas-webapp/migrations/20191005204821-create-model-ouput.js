'use strict';

module.exports = {
  up: (queryInterface, Sequelize) => {
    return queryInterface.createTable('ModelOutput',
     {
      Id: {
        allowNull: false,
        autoIncrement: true,
        primaryKey: true,
        type: Sequelize.INTEGER
      },
      ModelId: {
        type: Sequelize.INTEGER,
        references: {
          model: 'ModelRunDetail',
          key: 'Id'
        },
        onUpdate: 'cascade',
        onDelete: 'cascade'
      },
      BGPVal: Sequelize.FLOAT,
      WhiteSkedacityPval: Sequelize.FLOAT,
      VIFPval: Sequelize.FLOAT,
      ADFResidual: Sequelize.FLOAT,
      RSquared: Sequelize.FLOAT,
      RMSE: Sequelize.FLOAT,
      MAE: Sequelize.FLOAT,
      MAPE: Sequelize.FLOAT,
      AIC: Sequelize.FLOAT,
      DynamicBacktestRange1MAPE: Sequelize.FLOAT,
      DynamicBacktestRange2MAPE: Sequelize.FLOAT,
      DynamicBacktestRange3MAPE: Sequelize.FLOAT,
      DynamicBacktestRange4MAPE: Sequelize.FLOAT,
      DynamicBacktestRange5MAPE: Sequelize.FLOAT,
      DynamicBacktestRange6MAPE: Sequelize.FLOAT,
      DynamicBacktestRange7MAPE: Sequelize.FLOAT,
      DynamicBacktestRange8MAPE: Sequelize.FLOAT,
      DynamicBacktestRange9MAPE: Sequelize.FLOAT,
      DynamicBacktestRange10MAPE: Sequelize.FLOAT,
      DynamicBacktestLongRange1MAPE: Sequelize.FLOAT,
      DynamicBacktestLongRange2MAPE: Sequelize.FLOAT,
      DynamicBacktestLongRange3MAPE: Sequelize.FLOAT,
      DynamicBacktestLongRange4MAPE: Sequelize.FLOAT,
      DynamicBacktestLongRange5MAPE: Sequelize.FLOAT,
      DynamicBacktestLongRange6MAPE: Sequelize.FLOAT,
      DynamicBacktestLongRange7MAPE: Sequelize.FLOAT,
      DynamicBacktestLongRange8MAPE: Sequelize.FLOAT,
      DynamicBacktestLongRange9MAPE: Sequelize.FLOAT,
      DynamicBacktestLongRange10MAPE: Sequelize.FLOAT,
      DurbinWatson1: Sequelize.FLOAT,
      DurbinWatson2: Sequelize.FLOAT,
      DurbinWatson3: Sequelize.FLOAT,
      DurbinWatson4: Sequelize.FLOAT,
      AcceptReject: Sequelize.BOOLEAN,
      AcceptRejectReason: Sequelize.TEXT,
      ShapiroWilk: Sequelize.FLOAT,
      BreuschPagan: Sequelize.FLOAT,
      BreuschGodfrey: Sequelize.FLOAT
    });
  },

  down: (queryInterface, Sequelize) => {
    return queryInterface.dropTable('ModelOutput');
  }
};
